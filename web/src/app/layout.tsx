import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "@/components/Nav";

export const metadata: Metadata = {
  title: "AeroDoppler",
  description: "Consent-based aircraft-Doppler acoustic geolocation — local research tool",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Nav />
        <main className="container">{children}</main>
        <footer className="footer">
          AeroDoppler runs locally. Nothing is uploaded. Results are ranked hypotheses with <b>uncalibrated relative fit scores</b>, not verified locations. Flight data © adsb.lol feeders (ODbL 1.0) when the ADSB.lol provider is used.
        </footer>
      </body>
    </html>
  );
}
