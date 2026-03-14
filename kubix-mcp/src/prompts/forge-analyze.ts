import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';

export function registerKubixAnalyzePrompt(server: McpServer): void {
  server.registerPrompt(
    'kubix-analyze',
    {
      title: 'Analyze Code for GPU Optimization Opportunities',
      description:
        'Scan Python code for PyTorch operations that could benefit from Kubix GPU kernel optimization.',
      argsSchema: {
        code: z.string().describe('Python code to analyze'),
        context: z
          .string()
          .optional()
          .describe('Additional context about the project or performance needs'),
      },
    },
    ({ code, context }) => ({
      messages: [
        {
          role: 'user',
          content: {
            type: 'text',
            text: `You are an expert GPU performance analyst. Analyze the following code for GPU kernel optimization opportunities using the Kubix service.

CODE:
\`\`\`python
${code}
\`\`\`
${context ? `\nCONTEXT: ${context}` : ''}

Categorize each opportunity by impact:

**HIGH IMPACT** (recommend immediate Kubix optimization):
- Custom autograd functions (torch.autograd.Function) with manual forward/backward
- Attention mechanisms (especially without flash attention)
- Custom loss functions with reduction operations
- Fused operation opportunities (e.g., LayerNorm + activation, bias + GELU)
- Operations on large tensors (batch matmul, large convolutions)
- Performance-critical loops over tensor operations

**MEDIUM IMPACT** (worth optimizing if performance matters):
- Standard nn.Module operations that could be fused together
- Custom activation functions
- Normalization layers (BatchNorm, LayerNorm, RMSNorm, GroupNorm)
- Repeated small operations that could be batched into one kernel

**LOW IMPACT** (marginal gains):
- Simple element-wise operations on small tensors
- Operations already using optimized backends (cuDNN convolutions, cuBLAS matmul)
- I/O-bound operations (data loading, preprocessing)

For each opportunity:
1. Identify the exact code region (function/class name, line range)
2. Explain WHY it's an optimization target
3. Estimate potential speedup range (conservative to optimistic)
4. Show how to extract the code for kubix_optimize
5. Offer to run kubix_optimize on the best candidates immediately`,
          },
        },
      ],
    })
  );
}
