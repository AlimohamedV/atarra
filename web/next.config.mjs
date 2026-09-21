/** @type {import('next').NextConfig} */
import { fileURLToPath } from "node:url";
import { dirname } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));

const nextConfig = {
  reactStrictMode: true,
  // There is an unrelated lockfile higher up the drive, which makes Next.js guess
  // the wrong workspace root and emit a warning. Pin it to this directory.
  outputFileTracingRoot: here,
  // The API is consumed client-side, so the browser talks to it directly and no
  // server-side rewrite or proxy is needed.
};

export default nextConfig;
