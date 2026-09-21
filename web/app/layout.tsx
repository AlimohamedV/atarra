import type { Metadata } from "next";
// MapLibre's map controls need their own stylesheet. Importing it here (rather
// than in the client map component) keeps global CSS in the root layout, which is
// where the App Router expects it.
import "maplibre-gl/dist/maplibre-gl.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "ATARRA — Aquatic Invasive Weed Monitoring",
  description:
    "Multispectral satellite monitoring of Phragmites australis across Egypt's canals and coastal lagoons.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-100 antialiased">{children}</body>
    </html>
  );
}
