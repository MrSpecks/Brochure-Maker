## AI Brochure Maker (Streamlit)

Create a polished company brochure draft from a website. The app scrapes the site, filters for relevant pages (e.g., About, Careers), and uses an LLM (via OpenRouter) to produce a well-structured Markdown brochure.

---

### Impact
This project demonstrates the power of combining web scraping, intelligent link filtering, and Al-driven content generation to deliver ready-to-use marketing assets. It shows how businesses can reduce costs, speed up content creation, and maintain consistent messaging with minimal human input.

---

### 🛠️ Tech Stack
- **Python**
- **Streamlit** (app UI)
- **Libraries:**
  - `requests` – fetch web content
  - `openai` – connect to LLM via OpenRouter
  - `beautifulsoup4` – HTML parsing

---

### Features
- Scrapes landing page and finds relevant links (About, Careers, Team, Values, etc.)
- **Caches network calls** to keep the app fast
- **Two-column layout:** status/raw data vs. final brochure
- **Friendly error handling** and progress spinners
- **Website Scraping**: Fetches company web pages with proper request headers.
- **Content Filtering**: Identifies relevant links such as About, Company, or Careers.
- **AI-Powered Summarization**: Uses an LLM (via OpenAI/OpenRouter API) to condense content into brochure-ready text.
- **Structured Output**: Formats results as Markdown for brochure sections.
- **Interactive App**: Streamlit UI for step-by-step brochure creation and refinement.

---

### Setup

#### 1) Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2) Configure Secrets
Create `.streamlit/secrets.toml` and add your OpenRouter API key:
```toml
Openrouter_Api_Key="sk-or-..."
```

Alternatively, you can set the environment variable `OPENROUTER_API_KEY`.

#### 3) Environment Variables
- OPENAI_API_KEY (optional; not required if using OpenRouter via secrets)

#### 4) Run Locally
```bash
streamlit run streamlit_app.py
```

Open the printed local URL in your browser.

---

### Deployment
- Push this repository to GitHub
- On Streamlit Community Cloud, create a new app, point it to your repo, and set up `.streamlit/secrets.toml` with `Openrouter_Api_Key`
- Deploy. The app will use secrets in the cloud environment

---

### Notes
- Be sure the target site allows scraping and respect robots.txt and legal terms
- If few relevant links are found, try more specific subpages or different domains

---


### Example Workflow
- **Input:** `https://examplecompany.com`
- **Extracted:** About Us, Services, Careers pages
- **Output:** A company brochure summarizing mission, values, offerings, and opportunities.

---

### 🔮 Future Improvements
- Add image/logo scraping for branding.
- Export brochures as PDF or Word documents.
- Support multi-language brochure generation.

