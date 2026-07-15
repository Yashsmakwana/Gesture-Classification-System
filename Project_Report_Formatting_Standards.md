# Project Report Formatting & Structural Standards

This document outlines the universally accepted structure, formatting standards, and citation formats for academic and professional engineering/computer science project reports. It also provides a **Gap Analysis** comparing these standards against the system's current report ([Gesture_Classification_System_Report_Final.docx](file:///D:/Projects/Gesture-Classification-System/Gesture_Classification_System_Report_Final.docx)).

---

## 1. Standard Project Report Structure

An engineering or computer science project report is divided into three primary regions: **Front Matter**, **Main Body (Chapters)**, and **Back Matter**.

```
Project Report
├── Front Matter (Roman Numerals: i, ii, iii...)
│   ├── Title/Cover Page
│   ├── Certificate of Authenticity (Signed by Supervisor)
│   ├── Abstract & Keywords
│   ├── Table of Contents
│   ├── List of Figures
│   └── List of Tables
├── Main Body (Arabic Numerals: 1, 2, 3...)
│   ├── Chapter 1: Introduction (Background, Objectives, Scope)
│   ├── Chapter 2: Literature Review (Prior Art, Gaps, Justification)
│   ├── Chapter 3: System Requirements & Design (Hardware/Software, UML, Flowcharts)
│   ├── Chapter 4: Methodology & Preprocessing (Algorithms, Math Formulas)
│   ├── Chapter 5: Implementation (Code modules, execution flow)
│   ├── Chapter 6: Testing & Experimental Results (Metrics, confusion matrices, curves)
│   └── Chapter 7: Future Work & Conclusion (Azure Cloud proposal, summary)
└── Back Matter (Arabic Numerals Continued)
    ├── References (IEEE, APA, or Harvard style)
    └── Appendices (Installation manuals, full CSV schemas, raw outputs)
```

---

## 2. Document Typography & Layout Rules

To ensure readability and compliance with academic/binding regulations, the following standards are universally expected in PDF/Word report submissions:

| Attribute | Standard Specification | Rationale / Detail |
| :--- | :--- | :--- |
| **Page Size** | A4 (210mm x 297mm) | Global academic print standard. |
| **Margins** | Left: **1.5 in** (38mm)<br>Right, Top, Bottom: **1.0 in** (25.4mm) | Left margin is wider to allow room for spiral/hard book binding. |
| **Font Family** | Times New Roman or Calibri | Professional serif or clean sans-serif. |
| **Body text** | 12 pt size, **1.5 line spacing**, Justified | Enhances readability; 1.5 spacing is standard for draft reviews. |
| **Main Headings** | 14 pt or 16 pt, **Bold**, Title Case | e.g., **1. Introduction** (Always start chapters on a new page). |
| **Subheadings** | 12 pt or 13 pt, **Bold**, Left-Aligned | e.g., **3.1 Specific Objectives** |
| **Figure Captions** | 10 pt, Centered, **Below the figure** | e.g., *Figure 4. Training accuracy and loss curves.* |
| **Table Captions** | 10 pt, Left-Aligned, **Above the table** | e.g., *Table 2. Landmark Name and Description Mapping.* |

---

## 3. Academic Citation Standards

Citing external libraries, documentations, and publications is a strict requirement. The two most common formats in Computer Science are **IEEE** (numbered) and **APA** (author-date).

### A. IEEE Style (Recommended for Computer Science)
Citations are numbered in square brackets `[1]` in the order they appear in the text.
*   **Book/Manual Format:** Author(s), *Title*, edition, Publisher, Year.
*   **Web Reference Format:** Author(s), "Document Title," *Website*, Year. [Online]. Available: URL. [Accessed: Date].
*   **Examples:**
    *   `[1]` Google Developers, "MediaPipe Hand Landmarker Guide," *Google AI Edge*, 2026. [Online]. Available: https://ai.google.dev/edge/mediapipe. [Accessed: 15-Jul-2026].
    *   `[2]` K. Takahashi, "Hand Gesture Recognition using MediaPipe," *GitHub Repository*, 2020. [Online]. Available: https://github.com/KazuhitoTakahashi/hand-gesture-recognition-using-mediapipe. [Accessed: 15-Jul-2026].

### B. APA Style (Common in Interdisciplinary Sciences)
Citations use parenthetical author-date format: `(Google Developers, 2026)`.
*   **Format:** Author, A. A. (Year). *Title of document*. Publisher. URL
*   **Example:** 
    *   Google Developers. (2026). *MediaPipe Hand Landmarker Guide*. Google AI Edge. Retrieved from https://ai.google.dev/edge/mediapipe

---

## 4. Gap Analysis & Recommendations for Your Current Report

The current report `Gesture_Classification_System_Report_Final.docx` is highly structured and professional, but there are a few academic gaps. Addressing them will elevate the paper to a thesis-grade submission.

### Gap 1: Missing Literature Review / Technology Justification
*   **Observation:** The report explains "Need for the Project" (Section 2) but lacks a comparative review of alternative technologies.
*   **Recommendation:** Add a subsection under Chapter 2 comparing **MediaPipe (CPU keypoint-based)** against:
    *   *Raw CNNs (Convolutional Neural Networks):* Highly accurate but require GPUs (not suitable for real-time edge CPU).
    *   *OpenCV Contour/Color-based tracking:* Lightweight but highly sensitive to lighting, skin tone, and complex backgrounds.
    *   *Conclusion:* MediaPipe provides the perfect balance of robust skeleton extraction on standard CPUs.

### Gap 2: Informal Reference Formatting
*   **Observation:** The current "References" list contains plain text lines without standardized styling or access dates.
*   **Recommendation:** Reformat the references to follow the **IEEE standard** (as shown in Section 3 above). Ensure you add "Accessed: [Date]" for all online URLs.

### Gap 3: Missing Test Methodology / Test Cases
*   **Observation:** While Section 8 presents validation accuracy, confusion matrices, and metrics, it does not outline the software testing process.
*   **Recommendation:** Include a subsection detailing **Verification & Testing Methods**:
    *   *Unit Testing:* Testing landmark preprocessing math (scale/translation limits).
    *   *Integration Testing:* Testing coordinate logging to CSV and real-time capture overlay synchronization.
    *   *Boundary Testing:* What happens when no hands are in the frame (verifying fallback code `point_history.append([0, 0])` executes safely without throwing exceptions).
