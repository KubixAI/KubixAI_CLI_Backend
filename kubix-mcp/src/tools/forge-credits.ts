import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { getCredits } from '../api/kubix-client.js';
import { sanitizeError } from '../types.js';

export function registerKubixCredits(server: McpServer): void {
  server.registerTool(
    'kubix_credits',
    {
      title: 'Check Kubix Credits',
      description: `Check your current Kubix credit balance. Each single-GPU optimization costs 1 credit.
Credits are only charged for successful optimizations (speedup >= 1.1x).
Credits are lifetime (not monthly) and can be purchased on the dashboard.`,
      inputSchema: {},
    },
    async () => {
      try {
        const credits = await getCredits();
        return {
          content: [
            {
              type: 'text',
              text: [
                `## Kubix Credits`,
                `- **Balance**: ${credits.balance} credits`,
                `- **Total purchased**: ${credits.total_purchased}`,
                `- **Total used**: ${credits.total_used}`,
                `- **Plan**: ${credits.plan_type}`,
                `- **Email**: ${credits.email}`,
                '',
                credits.balance === 0
                  ? '⚠ No credits remaining. Purchase more at https://dashboard.kubixai.co/billing'
                  : `You can run ${credits.balance} more optimization(s).`,
              ].join('\n'),
            },
          ],
        };
      } catch (err) {
        return {
          content: [{ type: 'text', text: `Failed to check credits: ${sanitizeError(err)}` }],
          isError: true,
        };
      }
    }
  );
}
