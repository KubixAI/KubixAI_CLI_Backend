import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { getCredits } from '../api/kubix-client.js';

export function registerCreditsResource(server: McpServer): void {
  server.registerResource(
    'credits',
    'kubix://credits',
    {
      title: 'Kubix Credit Balance',
      description: 'Current Kubix credit balance and usage',
      mimeType: 'application/json',
    },
    async (uri) => {
      try {
        const credits = await getCredits();
        return {
          contents: [
            {
              uri: uri.href,
              mimeType: 'application/json',
              text: JSON.stringify({
                balance: credits.balance,
                total_purchased: credits.total_purchased,
                total_used: credits.total_used,
                plan_type: credits.plan_type,
                email: credits.email,
              }),
            },
          ],
        };
      } catch {
        return {
          contents: [
            {
              uri: uri.href,
              mimeType: 'application/json',
              text: JSON.stringify({
                error: 'Not authenticated. Use kubix_auth tool first.',
              }),
            },
          ],
        };
      }
    }
  );
}
