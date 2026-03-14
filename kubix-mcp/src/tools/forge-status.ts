import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { getJobStatus } from '../api/kubix-client.js';
import { sanitizeError } from '../types.js';

export function registerKubixStatus(server: McpServer): void {
  server.registerTool(
    'kubix_status',
    {
      title: 'Check Optimization Status',
      description:
        'Check the current status of a running or completed Kubix optimization job.',
      inputSchema: {
        session_id: z
          .string()
          .describe('The session ID returned from kubix_optimize or kubix_generate'),
      },
    },
    async ({ session_id }) => {
      try {
        const status = await getJobStatus(session_id);

        const lines = [
          `## Job Status: ${status.status.toUpperCase()}`,
          '',
          `| Field | Value |`,
          `|-------|-------|`,
          `| **Session** | ${status.session_id} |`,
          `| **Status** | ${status.status} |`,
          `| **Iteration** | ${status.current_iteration}/${status.max_iterations} |`,
          `| **Best Speedup** | ${status.best_speedup.toFixed(2)}x |`,
          `| **Best Iteration** | ${status.best_iteration} |`,
        ];

        if (status.error) {
          lines.push(`| **Error** | ${status.error} |`);
        }

        return { content: [{ type: 'text', text: lines.join('\n') }] };
      } catch (err) {
        return {
          content: [{ type: 'text', text: `Failed to get status: ${sanitizeError(err)}` }],
          isError: true,
        };
      }
    }
  );
}
