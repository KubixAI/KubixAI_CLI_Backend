import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { cancelJob } from '../api/kubix-client.js';
import { sanitizeError } from '../types.js';

export function registerKubixCancel(server: McpServer): void {
  server.registerTool(
    'kubix_cancel',
    {
      title: 'Cancel Optimization',
      description: 'Cancel a running Kubix optimization job. Credits may be refunded if less than 20% of iterations completed.',
      inputSchema: {
        session_id: z
          .string()
          .describe('The session ID of the job to cancel'),
      },
    },
    async ({ session_id }) => {
      try {
        await cancelJob(session_id);
        return {
          content: [
            {
              type: 'text',
              text: `Job ${session_id} has been cancelled. If less than 20% of iterations completed, credits will be refunded.`,
            },
          ],
        };
      } catch (err) {
        return {
          content: [{ type: 'text', text: `Failed to cancel job: ${sanitizeError(err)}` }],
          isError: true,
        };
      }
    }
  );
}
