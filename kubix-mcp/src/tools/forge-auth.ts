import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { loadTokens } from '../auth/token-store.js';
import { startOAuthFlow, validateTokens } from '../auth/oauth-client.js';
import { sanitizeError } from '../types.js';

export function registerKubixAuth(server: McpServer): void {
  server.registerTool(
    'kubix_auth',
    {
      title: 'Authenticate with Kubix',
      description: `Authenticate with the Kubix Kubix GPU kernel optimization service.
Opens the user's browser to sign in via the Kubix dashboard. Required before using any other Kubix tools.
If valid tokens already exist, verifies them without opening the browser.`,
      inputSchema: {
        force: z
          .boolean()
          .optional()
          .describe('Force re-authentication even if tokens exist'),
      },
    },
    async ({ force }) => {
      try {
        // Check existing tokens first (unless forced)
        if (!force) {
          const existing = loadTokens();
          if (existing) {
            const credits = await validateTokens();
            if (credits) {
              return {
                content: [
                  {
                    type: 'text',
                    text: `Already authenticated as ${credits.email} (${credits.plan_type} plan).\nKubix credits: ${credits.balance} available.`,
                  },
                ],
              };
            }
          }
        }

        // Start OAuth flow
        const tokens = await startOAuthFlow();

        // Validate and get user info
        const credits = await validateTokens();
        const email = credits?.email || 'unknown';
        const plan = credits?.plan_type || 'unknown';
        const balance = credits?.balance ?? 'unknown';

        return {
          content: [
            {
              type: 'text',
              text: `Successfully authenticated as ${email} (${plan} plan).\nKubix credits: ${balance} available.\nToken expires in ${Math.round(tokens.expires_in / 60)} minutes (auto-refreshes).`,
            },
          ],
        };
      } catch (err) {
        return {
          content: [{ type: 'text', text: `Authentication failed: ${sanitizeError(err)}` }],
          isError: true,
        };
      }
    }
  );
}
