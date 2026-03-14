import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { startOptimization, streamOptimization } from '../api/kubix-client.js';
import {
  sanitizeError,
  isIterationCompleteData,
  isSessionCompleteData,
  isErrorData,
} from '../types.js';
import type { SSEEvent } from '../types.js';
import { MAX_PYTORCH_CODE_BYTES, MAX_USER_PROMPT_BYTES } from '../constants.js';

export function registerKubixOptimize(server: McpServer): void {
  server.registerTool(
    'kubix_optimize',
    {
      title: 'Optimize GPU Kernel',
      description: `Submit PyTorch code to Kubix for GPU kernel optimization.
Kubix generates optimized Triton or CUDA kernels, evaluates them on real H100/A100 GPUs,
and returns the best result with speedup metrics over torch.compile(max-autotune).

USE THIS TOOL WHEN YOU SEE:
- PyTorch custom operations (torch.autograd.Function, custom forward/backward)
- Manual CUDA kernels that could be faster
- Performance-critical tensor operations (attention, convolution, normalization, softmax)
- Code with comments like "slow", "bottleneck", "optimize", "performance"
- torch.compile() targets or triton.jit kernels
- Any nn.Module with significant compute in forward()
- Matrix multiplication, reduction, or scan operations
- Custom loss functions with reduction operations
- Fused operation opportunities (e.g., LayerNorm + activation)

The tool streams real-time progress and blocks until optimization completes (1-10 minutes).
Cost: 1 credit per optimization (only charged if speedup >= 1.1x).
Requires authentication via kubix_auth first.`,
      inputSchema: {
        pytorch_code: z
          .string()
          .max(MAX_PYTORCH_CODE_BYTES)
          .describe(
            'Complete PyTorch code to optimize. Must be a runnable module with a model class.'
          ),
        kernel_name: z
          .string()
          .max(200)
          .describe(
            'Short descriptive name for this kernel (e.g., "flash_attention", "fused_layernorm")'
          ),
        output_format: z
          .enum(['triton', 'native_cuda'])
          .default('triton')
          .describe(
            'Output format. "triton" is recommended (portable, readable). "native_cuda" for maximum control.'
          ),
        target_speedup: z
          .number()
          .min(1.0)
          .max(100.0)
          .default(2.0)
          .describe('Target speedup multiplier over torch.compile baseline'),
        max_iterations: z
          .number()
          .min(1)
          .max(100)
          .default(10)
          .describe('Maximum optimization iterations. More iterations = higher chance of better speedup.'),
        gpu: z
          .enum(['B200', 'H200', 'H100', 'L40S', 'L4', 'A100', 'A10', 'T4'])
          .default('H100')
          .describe('Target GPU for optimization and benchmarking'),
        user_prompt: z
          .string()
          .max(MAX_USER_PROMPT_BYTES)
          .optional()
          .describe(
            'Optional guidance for the optimizer (e.g., "focus on memory bandwidth", "use shared memory tiling")'
          ),
      },
    },
    async (args) => {
      try {
        // Start optimization
        const { session_id } = await startOptimization({
          pytorch_code: args.pytorch_code,
          kernel_name: args.kernel_name,
          output_format: args.output_format,
          target_speedup: args.target_speedup,
          max_iterations: args.max_iterations,
          gpu: args.gpu,
          user_prompt: args.user_prompt,
        });

        // Collect events from SSE stream
        const iterations: {
          num: number;
          speedup: number;
          correct: boolean;
          isBest: boolean;
        }[] = [];
        let bestCode = '';
        let bestSpeedup = 0;
        let sessionComplete: { best_speedup: number; best_iteration: number; latency_before?: number; latency_after?: number; optimized_code: string; termination_reason?: string } | undefined;
        let errorMessage: string | undefined;

        await streamOptimization(session_id, (event: SSEEvent) => {
          if (event.type === 'iteration_complete' && isIterationCompleteData(event.data)) {
            const d = event.data;
            iterations.push({
              num: d.iteration,
              speedup: d.speedup,
              correct: d.correct,
              isBest: d.is_best ?? false,
            });
            if (d.is_best && d.code) {
              bestCode = d.code;
              bestSpeedup = d.best_speedup;
            }
          } else if (event.type === 'session_complete' && isSessionCompleteData(event.data)) {
            sessionComplete = event.data;
          } else if (event.type === 'error' && isErrorData(event.data)) {
            errorMessage = event.data.message;
          }
        });

        if (errorMessage) {
          return {
            content: [
              {
                type: 'text',
                text: `## Kubix Optimization Failed\n\n**Session**: ${session_id}\n**Error**: ${errorMessage}`,
              },
            ],
            isError: true,
          };
        }

        const finalCode = sessionComplete?.optimized_code || bestCode;
        const finalSpeedup = sessionComplete?.best_speedup || bestSpeedup;
        const latencyBefore = sessionComplete?.latency_before;
        const latencyAfter = sessionComplete?.latency_after;
        const reason = sessionComplete?.termination_reason || 'completed';
        const correctCount = iterations.filter((i) => i.correct).length;

        const lines: string[] = [
          `## Kubix Optimization Results`,
          '',
          `| Metric | Value |`,
          `|--------|-------|`,
          `| **Kernel** | ${args.kernel_name} |`,
          `| **GPU** | ${args.gpu} |`,
          `| **Format** | ${args.output_format} |`,
          `| **Best Speedup** | ${finalSpeedup.toFixed(2)}x |`,
          `| **Iterations** | ${iterations.length}/${args.max_iterations} |`,
          `| **Correct Kernels** | ${correctCount}/${iterations.length} |`,
        ];

        if (latencyBefore != null && latencyAfter != null) {
          lines.push(
            `| **Latency** | ${latencyBefore.toFixed(2)}ms → ${latencyAfter.toFixed(2)}ms |`
          );
        }

        lines.push(`| **Termination** | ${reason} |`);
        lines.push(`| **Session ID** | ${session_id} |`);

        if (finalCode) {
          lines.push('', '### Optimized Kernel Code', '', '```python', finalCode, '```');
        }

        // Iteration history
        lines.push('', '### Iteration History', '');
        for (const it of iterations) {
          const mark = it.isBest ? ' ★' : '';
          const status = it.correct ? '✓' : '✗';
          lines.push(
            `- Iteration ${it.num}: ${it.speedup.toFixed(2)}x ${status}${mark}`
          );
        }

        if (finalSpeedup >= 1.1 && finalCode) {
          lines.push(
            '',
            '### Integration',
            '',
            'To use this optimized kernel, replace your PyTorch implementation with the generated code above.',
            'The optimized kernel was benchmarked against `torch.compile(mode="max-autotune")` and achieved',
            `a **${finalSpeedup.toFixed(2)}x speedup**.`
          );
        }

        return { content: [{ type: 'text', text: lines.join('\n') }] };
      } catch (err) {
        return {
          content: [{ type: 'text', text: `Optimization failed: ${sanitizeError(err)}` }],
          isError: true,
        };
      }
    }
  );
}
