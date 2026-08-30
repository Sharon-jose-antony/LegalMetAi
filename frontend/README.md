# LEGALMET AI — Frontend Portal

**AI-Assisted Legal Metrology Compliance Inspection Platform (SIH26034)**

The frontend is a modern, responsive Single-Page Application (SPA) designed for both desktop workstations and mobile smartphones used by field enforcement inspectors.

---

## 🚀 Features

- **Inspector Dashboard**: Enforcement statistics, pass rates, compliance distribution, and recent activity.
- **Quick Camera Capture**: Mobile camera live capture with automated high-resolution optimization and autofocus support.
- **Real-Time Image Quality Meter**: Instant feedback on blur (Laplacian variance), contrast, resolution, and specular glare.
- **Multi-Pass OCR Visualizer**: View bounding boxes and detected text tokens across multiple image enhancements.
- **Statutory Declaration Table**: Category-aware validation status chips (`FOUND`, `NOT_FOUND`, `NOT_APPLICABLE`, `UNCERTAIN`, `MANUAL_REVIEW`).
- **Evidence Crop Gallery**: High-resolution cropped image snippets for every detected declaration with lightbox zoom.
- **Manual Review & Override**: Inspector sign-off portal to confirm, override, or reject AI preliminary findings with statutory notes.
- **PDF Report Download**: Evidentiary PDF generation adhering to Ministry of Consumer Affairs guidelines.
- **Mobile QR / Local Network IP**: Direct Wi-Fi pairing for field mobile devices.

---

## 🛠 Running the Frontend

### Option 1: Served Automatically by FastAPI Backend (Recommended)
When the FastAPI backend is running, the frontend is served directly at the root URL:
```
http://localhost:8000/
```
No separate frontend server is required in this mode.

### Option 2: Standalone Static Server (Node.js)
If you wish to serve the frontend separately (e.g. on port 3000):

1. Install dependencies:
   ```bash
   npm install
   ```
2. Start the dev server:
   ```bash
   npm run dev
   ```
   or
   ```bash
   npm start
   ```
3. Open `http://localhost:3000` in your browser.

---

## 📱 Mobile Field Testing

To use the camera on a smartphone:
1. Connect your phone to the same local Wi-Fi network as the inspection host.
2. In the inspector dashboard, click the **Local Network IP** badge at the bottom of the sidebar.
3. Open the displayed URL (e.g., `http://192.168.1.X:8000`) in your mobile browser.
4. Tap **📸 Capture Product** — your phone's native camera will open automatically.

---

## ⚙ Environment Configuration

Copy `.env.example` to `.env` if using a build system:
```env
VITE_API_URL=http://localhost:8000
```
