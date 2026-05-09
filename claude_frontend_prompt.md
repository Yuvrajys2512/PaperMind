# PaperMind Frontend UI Redesign Prompt

**Role:** You are an expert Frontend Engineer and UI/UX Designer specializing in high-end, premium web experiences.

**Task:** Completely revamp the PaperMind React/Vite/Tailwind v4 frontend to make it visually stunning, highly dynamic, and premium. The current UI is too basic and generic. We want a design that breaks away from the typical "AI slop" and feels like a futuristic, sophisticated data visualization platform. 

**Reference Aesthetic:** The target design should be "Cosmic Glassmorphism." Think deep space/dark mode backgrounds, vibrant neon glows (purples, pinks, cyan, and deep blues), semi-transparent glass panels with soft glowing borders, and modern typography. 

### 🛑 CRITICAL CONSTRAINTS (READ FIRST)
1. **NO BACKEND CHANGES:** You are strictly forbidden from modifying any Python files, API routes, `api/`, `ingestion/`, or backend logic.
2. **NO LOGIC CHANGES:** The core React state, API fetching logic, and data structures must remain EXACTLY the same. Do not break the current flow for uploading PDFs, asking questions, displaying answers, or evidence grading.
3. **UI/UX ONLY:** Your changes should only target styling (`.css`), component rendering (JSX), animations, and the addition of non-functional visual placeholders.

### 🎨 Design & Animation Requirements
- **Theme:** Implement a deep, immersive dark theme with cosmic accents. Use layered backgrounds, subtle radial gradients, and glassmorphic panels (backdrop blur, low opacity backgrounds, delicate borders).
- **Animations:** Integrate subtle, high-quality micro-animations. Elements should smoothly fade/slide in. Use CSS animations, Framer Motion, or similar for hover states, expanding accordions (like the Evidence Breakdown), and loading states. Add a slow, ambient moving element in the background (like subtle floating particles or shifting gradient meshes).
- **Data Visualizations:** The UI needs to look data-rich. Add visually appealing, animated decorative graphs (e.g., glowing sine waves, sparklines, or circular progress rings) to the dashboard and chat areas. These can be dummy visualizers that react to the "confidence" scores or just provide a high-tech ambiance.
- **Typography:** Use a modern, clean font (e.g., Inter, Outfit, or Space Grotesk). Ensure high readability despite the dark theme.

### 🔮 Future Feature Placeholders
We have an ongoing implementation plan. You must build beautiful UI placeholders/shells for the following upcoming features so the UI is ready for them:
1. **Table Extraction (Session 5):** Add a visual indicator or chip in the source references for "Table Extracted". Create a sleek, glowing modal or expandable panel placeholder for viewing extracted markdown tables.
2. **Structured Logging & Tracing (Session 6):** Add a high-tech "Telemetry / Request Trace" panel (can be a drawer or a sidebar section). It should have placeholder metrics for request ID, timing breakdown (ms), and pipeline stages, looking like a developer or diagnostics HUD.
3. **Multi-paper Comparison (Session 7):** In the workspace/PDF selector area, add a "Compare Mode" toggle switch. When toggled on, show a beautifully designed dual-selector placeholder allowing the user to select two different papers side-by-side.

### 🛠️ Execution Steps
1. **Analyze:** Review `frontend/src/pages/ChatPage.jsx`, `UploadPage.jsx`, and `index.css` to understand the current structure.
2. **Setup Styles:** Overhaul `index.css` or Tailwind config to introduce the new cosmic color palette, glassmorphism utilities, and custom animations.
3. **Refactor Components:** Apply the new styles to all existing components. Ensure the Evidence Grading UI (DIRECT/INFERRED/REMOVED chips) looks incredibly premium with glowing dots and smooth expansion.
4. **Add Visuals & Placeholders:** Inject the ambient background animations, decorative graphs, and the UI placeholders for Tables, Telemetry, and Compare Mode.
5. **Verify:** Ensure no API calls or React state logic were broken during the styling process.

Make it look incredibly polished, dynamic, and breathtaking!
