import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';

export function registerKubixOptimizePrompt(server: McpServer): void {
  server.registerPrompt(
    'kubix-optimize',
    {
      title: 'Optimize GPU Kernel with Kubix',
      description:
        'Analyze PyTorch code and optimize it into a faster Triton/CUDA kernel using the Kubix service.',
      argsSchema: {
        code: z.string().describe('PyTorch code to optimize'),
        goal: z
          .string()
          .optional()
          .describe(
            'Optimization goal (e.g., "reduce memory usage", "maximize throughput", "lower latency")'
          ),
      },
    },
    ({ code, goal }) => ({
      messages: [
        {
          role: 'user',
          content: {
            type: 'text',
            text: `You are an expert GPU kernel optimization assistant with access to the Kubix optimization service.

TASK: Analyze the following PyTorch code and optimize it using the kubix_optimize tool.

CODE:
\`\`\`python
${code}
\`\`\`
${goal ? `\nOPTIMIZATION GOAL: ${goal}` : ''}

INSTRUCTIONS:
1. First, call kubix_credits to check if the user has credits available.
2. Analyze the code to identify performance-critical operations.
3. Call kubix_optimize with the code, choosing appropriate parameters:
   - kernel_name: A short descriptive name for the operation
   - output_format: "triton" (recommended for portability) or "native_cuda" (maximum control)
   - gpu: "H100" for best performance
   - target_speedup: 2.0 is a good default
   - max_iterations: 10 for thorough optimization
4. While waiting for results, explain what Kubix is doing:
   - LLM-guided search generates candidate kernels
   - Each kernel is validated for correctness
   - Kernels are benchmarked on real H100 GPUs against torch.compile(max-autotune)
   - The search iterates to find faster implementations
5. When results arrive, present the optimized kernel and explain:
   - The speedup achieved over torch.compile
   - Key optimizations (tiling, memory coalescing, shared memory, kernel fusion)
   - How to integrate the kernel into the user's codebase
   - Any caveats (numerical precision, supported shapes, etc.)`,
          },
        },
      ],
    })
  );
}
