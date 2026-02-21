import { mkdir } from "node:fs/promises";
import path from "node:path";
import { readdir, rename, copyFile, writeFile } from "node:fs/promises";

export const SessionFolderPlugin = async ({ directory, client }) => {
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const baseDir = path.resolve(directory, ".opencode_sessions");
  const sessionDir = path.join(baseDir, timestamp);
  
  try {
    await mkdir(baseDir, { recursive: true });
    await mkdir(sessionDir, { recursive: true });
    
    // Create a link file to track the session
    const linkFile = path.join(baseDir, "current_session.txt");
    await writeFile(linkFile, sessionDir, "utf8");
    
    // Store the session directory globally
    global._currentSessionDir = sessionDir;
    global._sessionBaseDir = baseDir;
    global._sessionTimestamp = timestamp;
    global._processedExports = new Set();
    
    await client.app.log({
      body: {
        service: "session-folder-plugin",
        level: "info",
        message: `Session folder: ${timestamp}`
      }
    });
    
    // Start polling for export files
    const pollInterval = setInterval(async () => {
      try {
        const files = await readdir(directory);
        
        for (const file of files) {
          // Check if it's a session export file we haven't processed
          if (file.startsWith('session-') && file.endsWith('.md') && !global._processedExports.has(file)) {
            global._processedExports.add(file);
            
            const sourcePath = path.join(directory, file);
            const destPath = path.join(sessionDir, file);
            
            await client.app.log({
              body: {
                service: "session-folder-plugin",
                level: "info",
                message: `Found export: ${file}`
              }
            });
            
            try {
              await rename(sourcePath, destPath);
              await client.app.log({
                body: {
                  service: "session-folder-plugin",
                  level: "info",
                  message: `Moved to session folder`
                }
              });
            } catch (e) {
              try {
                await copyFile(sourcePath, destPath);
                await client.app.log({
                  body: {
                    service: "session-folder-plugin",
                    level: "info",
                    message: `Copied to session folder`
                  }
                });
              } catch (e2) {
                await client.app.log({
                  body: {
                    service: "session-folder-plugin",
                    level: "error",
                    message: `Failed: ${e2.message}`
                  }
                });
              }
            }
          }
        }
      } catch (e) {
        // Silent fail on polling errors
      }
    }, 1000); // Check every second
    
    // Store interval to clean up later if needed
    global._exportPollInterval = pollInterval;
    
  } catch (error) {
    console.error("[SessionFolderPlugin] Error:", error);
  }
  
  return {};
};