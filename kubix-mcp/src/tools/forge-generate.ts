import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { startGeneration, streamOptimization } from '../api/kubix-client.js';
import {
  sanitizeError,
  isIterationCompleteData,
  isSessionCompleteData,
  isErrorData,
} from '../types.js';
import type { SSEEvent } from '../types.js';
import { MAX_USER_PROMPT_BYTES } from '../constants.js';

export function registerKubixGenerate(server: McpServer): void {
  server.registerTool(
    'kubix_generate',
    {
      title: 'Generate GPU Kernel',
      description: `Generate an optimized GPU kernel from scratch based on a specification.
Use this when you need to create a new high-performance kernel without existing PyTorch code.
Kubix will generate a PyTorch baseline first, then optimize it into Triton or CUDA.

Requires authentication via kubix_auth first. Cost: 1 credit per generation.`,
      inputSchema: {
        operation: z
          .string()
          .max(200)
          .describe(
            'Operation name (e.g., "fused_attention", "custom_gelu", "softmax", "matmul")'
          ),
        description: z
          .string()
          .max(MAX_USER_PROMPT_BYTES)
          .describe('Detailed description of what the kernel should do'),
        input_shapes: z
          .array(z.array(z.number()))
          .describe('Input tensor shapes, e.g., [[8, 512, 768]] for a single input'),
        output_shape: z
          .array(z.number())
          .optional()
          .describe('Expected output tensor shape'),
        dtype: z
          .string()
          .default('float16')
          .describe('Data type (float16, float32, bfloat16)'),
        output_format: z
          .enum(['triton', 'native_cuda'])
          .default('triton')
          .describe('Output format'),
        target_speedup: z.number().min(1.0).max(100.0).default(2.0),
        max_iterations: z.number().min(1).max(100).default(10),
        gpu: z
          .enum(['B200', 'H200', 'H100', 'L40S', 'L4', 'A100', 'A10', 'T4'])
          .default('H100'),
        user_prompt: z
          .string()
          .max(MAX_USER_PROMPT_BYTES)
          .optional(),
      },
    },
    async (args) => {
      try {
        const { session_id } = await startGeneration({
          operation: args.operation,
          description: args.description,
          input_shapes: args.input_shapes,
          output_shape: args.output_shape,
          dtype: args.dtype,
          output_format: args.output_format,
          target_speedup: args.target_speedup,
          max_iterations: args.max_iterations,
          gpu: args.gpu,
          user_prompt: args.user_prompt,
        });

        const iterations: {
          num: number;
          speedup: number;
          correct: boolean;
          isBest: boolean;
        }[] = [];
        let bestCode = '';
        let bestSpeedup = 0;
        let sessionComplete: { best_speedup: number; optimized_code: string } | undefined;
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
                text: `## Kubix Generation Failed\n\n**Session**: ${session_id}\n**Error**: ${errorMessage}`,
              },
            ],
            isError: true,
          };
        }

        const finalCode = sessionComplete?.optimized_code || bestCode;
        const finalSpeedup = sessionComplete?.best_speedup || bestSpeedup;
        const correctCount = iterations.filter((i) => i.correct).length;

        const lines: string[] = [
          `## Kubix Kernel Generation Results`,
          '',
          `| Metric | Value |`,
          `|--------|-------|`,
          `| **Operation** | ${args.operation} |`,
          `| **GPU** | ${args.gpu} |`,
          `| **Format** | ${args.output_format} |`,
          `| **Best Speedup** | ${finalSpeedup.toFixed(2)}x |`,
          `| **Iterations** | ${iterations.length}/${args.max_iterations} |`,
          `| **Correct Kernels** | ${correctCount}/${iterations.length} |`,
          `| **Session ID** | ${session_id} |`,
        ];

        if (finalCode) {
          lines.push('', '### Generated Kernel Code', '', '```python', finalCode, '```');
        }

        lines.push('', '### Iteration History', '');
        for (const it of iterations) {
          const mark = it.isBest ? ' ★' : '';
          const status = it.correct ? '✓' : '✗';
          lines.push(
            `- Iteration ${it.num}: ${it.speedup.toFixed(2)}x ${status}${mark}`
          );
        }

        return { content: [{ type: 'text', text: lines.join('\n') }] };
      } catch (err) {
        return {
          content: [{ type: 'text', text: `Generation failed: ${sanitizeError(err)}` }],
          isError: true,
        };
      }
    }
  );
}
