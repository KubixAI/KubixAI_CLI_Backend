"""
Kubix CLI - Async OpenRouter API Client

Improved API client with:
- Async/await for concurrent requests
- Exponential backoff retry logic
- Rate limiting
- Better error handling
- Request/response logging
"""

import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from ..utils.validation import APIConfig
from ..exceptions import (
    OpenRouterAPIError, APIKeyError,
    APIRateLimitError, APITimeoutError
)
from ..logger import get_logger, log_api_request, log_api_response


logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """Result from AI generation."""
    code: str
    model: str
    generation_time_ms: float
    tokens_used: int
    variant_index: int


class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire rate limit slot."""
        async with self._lock:
            now = time.time()
            time_since_last = now - self.last_call

            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)

            self.last_call = time.time()


class AsyncOpenRouterClient:
    """
    Async OpenRouter API client with retry logic and rate limiting.

    Features:
    - Async HTTP requests
    - Exponential backoff retry
    - Rate limiting
    - Timeout handling
    - Structured logging
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, config: APIConfig):
        """
        Initialize async API client.

        Args:
            config: API configuration
        """
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Kubix-AI/kubix-cli",
            "X-Title": "Kubix CLI"
        }
        self.rate_limiter = RateLimiter(config.rate_limit_per_minute)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
        return False

    async def generate_kernel_optimizations_async(
        self,
        original_code: str,
        analysis: Dict[str, Any],
        constraints: Dict[str, Any],
        num_variants: int = 3
    ) -> List[GenerationResult]:
        """
        Generate multiple optimization variants concurrently.

        Args:
            original_code: Original kernel code
            analysis: Kernel analysis
            constraints: Compilation constraints
            num_variants: Number of variants to generate

        Returns:
            List of GenerationResult
        """
        if not self._session:
            self._session = aiohttp.ClientSession()

        system_prompt = self._create_system_prompt()
        base_prompt = self._create_user_prompt(original_code, analysis, constraints)

        # Generate variants in parallel
        tasks = [
            self._generate_single_variant(
                system_prompt,
                base_prompt,
                variant_index=i
            )
            for i in range(num_variants)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        successful = [
            r for r in results
            if not isinstance(r, Exception) and r is not None
        ]

        return successful

    async def _generate_single_variant(
        self,
        system_prompt: str,
        base_prompt: str,
        variant_index: int
    ) -> Optional[GenerationResult]:
        """Generate a single optimization variant."""
        try:
            # Add variant-specific instruction
            prompt = f"{base_prompt}\n\nGenerate variant {variant_index + 1} with a unique optimization strategy."

            log_api_request(
                logger,
                model=self.config.model,
                prompt_length=len(prompt),
                variant=variant_index
            )

            start_time = time.time()

            # Make API request with retry logic
            response_text = await self._make_request_with_retry(
                system_prompt,
                prompt
            )

            generation_time = (time.time() - start_time) * 1000

            log_api_response(
                logger,
                model=self.config.model,
                response_length=len(response_text),
                duration_ms=generation_time
            )

            # Extract code from response
            code = self._extract_cuda_code(response_text)

            if code:
                return GenerationResult(
                    code=code,
                    model=self.config.model,
                    generation_time_ms=generation_time,
                    tokens_used=len(response_text.split()),  # Rough estimate
                    variant_index=variant_index
                )

            logger.warning(
                "Failed to extract code from response",
                variant=variant_index
            )
            return None

        except Exception as e:
            logger.error(
                "Variant generation failed",
                variant=variant_index,
                error=str(e)
            )
            return None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((
            aiohttp.ClientError,
            asyncio.TimeoutError,
            APIRateLimitError
        ))
    )
    async def _make_request_with_retry(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """
        Make API request with automatic retry logic.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt

        Returns:
            Response text

        Raises:
            OpenRouterAPIError: On API error
            APITimeoutError: On timeout
            APIRateLimitError: On rate limit (retried automatically)
        """
        # Apply rate limiting
        await self.rate_limiter.acquire()

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)

            async with self._session.post(
                self.BASE_URL,
                headers=self.headers,
                json=payload,
                timeout=timeout
            ) as response:

                # Handle rate limiting
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(
                        "Rate limited, retrying",
                        retry_after=retry_after
                    )
                    raise APIRateLimitError(retry_after)

                # Handle auth errors
                if response.status == 401:
                    raise APIKeyError("Invalid OpenRouter API key")

                # Handle other errors
                if response.status >= 400:
                    body = await response.text()
                    raise OpenRouterAPIError(
                        response.status,
                        f"API request failed: {body[:200]}",
                        body
                    )

                # Parse response
                data = await response.json()
                return data["choices"][0]["message"]["content"]

        except asyncio.TimeoutError:
            raise APITimeoutError(self.config.timeout_seconds)

        except aiohttp.ClientError as e:
            logger.error("HTTP client error", error=str(e))
            raise

    def _create_system_prompt(self) -> str:
        """Create system prompt for kernel optimization."""
        return """You are an expert CUDA performance engineer specializing in kernel optimization.

Your task is to optimize CUDA kernels by:
1. Analyzing the current implementation for bottlenecks
2. Applying advanced optimization techniques
3. Ensuring correctness is maintained
4. Maximizing performance for the target GPU

Key optimization strategies:
- Memory coalescing and access patterns
- Shared memory utilization and bank conflict avoidance
- Warp-level primitives and shuffle operations
- Instruction-level parallelism and loop unrolling
- Register usage optimization
- Minimizing thread divergence
- Using vectorized loads/stores (float2, float4)
- Tensor Core utilization where applicable
- Optimal launch configuration

Always provide complete, compilable CUDA code with:
- Clear comments explaining optimizations
- Proper error checking
- Expected performance improvements

Return ONLY the optimized CUDA kernel code wrapped in ```cuda code blocks."""

    def _create_user_prompt(
        self,
        code: str,
        analysis: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """Create user prompt for optimization."""
        return f"""Optimize this CUDA kernel for better performance:

```cuda
{code}
```

Kernel Analysis:
- Name: {analysis.get('kernel_name', 'unknown')}
- Current patterns: {', '.join(analysis.get('patterns', []))}
- Arithmetic intensity: {analysis.get('arithmetic_intensity', 0):.2f}
- Complexity: {analysis.get('complexity', 'unknown')}

Optimization Opportunities:
{chr(10).join(f"- {opp}" for opp in analysis.get('optimization_opportunities', []))}

Constraints:
- Max registers: {constraints.get('max_registers', 255)}
- Max shared memory: {constraints.get('shared_memory_kb', 48)}KB
- Target GPU: {constraints.get('target_gpu', 'sm_70')}

Focus on: memory coalescing, shared memory usage, instruction-level parallelism, and minimizing divergence.
Provide a complete, compilable CUDA kernel with detailed optimization comments."""

    def _extract_cuda_code(self, response: str) -> Optional[str]:
        """Extract CUDA code from API response."""
        import re

        # Try to extract from code blocks
        cuda_pattern = r'```(?:cuda|cpp|c\+\+)?\n(.*?)```'
        matches = re.findall(cuda_pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Fallback: look for __global__ keyword
        if "__global__" in response and "void" in response:
            lines = response.split('\n')
            code_lines = []
            in_code = False

            for line in lines:
                if "__global__" in line or "__device__" in line or "#include" in line:
                    in_code = True
                if in_code:
                    code_lines.append(line)

            if code_lines:
                return '\n'.join(code_lines).strip()

        return None

    async def validate_api_key_async(self) -> bool:
        """Validate API key with test request."""
        if not self._session:
            self._session = aiohttp.ClientSession()

        try:
            await self._make_request_with_retry(
                "You are a helpful assistant.",
                "Hello, this is a test."
            )
            return True
        except APIKeyError:
            return False
        except Exception:
            return False

    def close(self):
        """Close the client session synchronously."""
        if self._session and not self._session.closed:
            asyncio.create_task(self._session.close())


# Synchronous wrapper for backward compatibility
class OpenRouterClientSync:
    """Synchronous wrapper around async client."""

    def __init__(self, api_key: str):
        config = APIConfig(api_key=api_key)
        self.async_client = AsyncOpenRouterClient(config)
        self._loop = None

    def generate_kernel_optimizations(
        self,
        original_code: str,
        analysis: Dict[str, Any],
        constraints: Dict[str, Any],
        num_variants: int = 3
    ) -> List[Any]:
        """Synchronous wrapper for generation."""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()

        async def _generate():
            async with self.async_client as client:
                return await client.generate_kernel_optimizations_async(
                    original_code, analysis, constraints, num_variants
                )

        results = self._loop.run_until_complete(_generate())

        # Convert to old format for compatibility
        from ..openrouter import KernelCandidate

        return [
            KernelCandidate(
                code=r.code,
                operation=analysis.get('kernel_name', 'unknown'),
                constraints=constraints,
                metadata={
                    'variant': r.variant_index,
                    'model': r.model,
                    'generation_time_ms': r.generation_time_ms,
                    'tokens_used': r.tokens_used
                }
            )
            for r in results
        ]

    def __del__(self):
        """Cleanup loop on deletion."""
        if self._loop and self._loop.is_running():
            self._loop.close()
