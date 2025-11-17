# research_agent.py

import re
from docx import Document
from openai import OpenAI, RateLimitError
import yfinance as yf
import streamlit as st

# OpenAI client reads api key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Simple cache to avoid repeated LLM calls for the same question
LLM_CACHE = {}


def _norm(text: str) -> str:
    text = text.replace("–", "-").replace("—", "-").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def _norm_question(q: str) -> str:
    q = q.strip().lower()
    return re.sub(r"\s+", " ", q)


# Expanded FAQ for greetings, meta questions and common company questions
FAQ_QA = {
    # greetings and meta
    "hello": "Hello, I am your research agent chatbot.",
    "hi": "Hello, I am your research agent chatbot.",
    "who are you": "I am a chatbot that answers using JPMorgan, Pfizer, and Google research documents, plus live market data for these companies.",
    "what can you do": "I answer questions based on JPMorgan, Pfizer, and Google research documents and I can also provide simple live market information such as share price and recent trend for these companies.",
    "which companies are covered": "I currently cover JPMorgan, Pfizer, and Google.",
    "what documents are you using": "I use deep dive research documents for JPMorgan, Pfizer, and Google that you loaded into the system, and for market questions I use live market data.",
    "how should i ask questions": "Mention the company name and the topic you care about, such as technology strategy, digital transformation, research and development pipeline, cloud, artificial intelligence, or market trend.",
    "what companies do you cover": "I currently cover JPMorgan, Pfizer, and Google using your uploaded research documents and live market data.",
    "how do you answer questions": "I answer using your research documents first, and if it is a market question, I use live stock data.",
    "how to use this chatbot": "Ask about Google, JPMorgan, or Pfizer strategy, technology, cloud, AI, digital transformation, pipeline, research and development, or stock performance.",
    "what topics can you answer": "AI, cloud, digital transformation, investment, engineering, infrastructure, pipeline, financial performance, market trend, and high level summaries.",

    # JPMorgan general
    "what is jpmorgan": (
        "JPMorgan is a global financial services firm with operations in retail banking, "
        "investment banking, asset management, and payments. The document describes a strong focus "
        "on technology, modern platforms, and digital services for clients."
    ),
    "what is jpmorgan main motive": (
        "JPMorgan main motive in the document is to strengthen its position as a leading global bank "
        "by investing heavily in technology, improving customer experience, modernizing core systems, "
        "and using data and artificial intelligence to run the business more efficiently and safely."
    ),
    "summarise jpmorgan document": (
        "The JPMorgan document describes how the bank is modernizing its technology estate, moving workloads "
        "to cloud, simplifying applications, improving data platforms, and using digital channels to serve clients. "
        "It highlights large annual technology spend and a big engineering workforce that supports payments, trading, "
        "risk, and customer facing products."
    ),

    # JPMorgan strategy
    "what is jpmorgan technology strategy": (
        "JPMorgan technology strategy is to invest at scale in modern platforms, cloud adoption, and resilient "
        "infrastructure while simplifying legacy systems. The bank wants technology that is secure, reliable, "
        "and flexible so it can deliver new digital products quickly, use data more effectively, and support global growth."
    ),
    "what is jpmorgan digital transformation strategy": (
        "JPMorgan digital transformation strategy focuses on building modern digital experiences for clients, "
        "streamlining operations, and using data and analytics to improve decisions. The bank is upgrading applications, "
        "moving to cloud, and adopting modern engineering practices so that innovation can be delivered faster and at scale."
    ),
    "what is jpmorgan ai strategy": (
        "In the document JPMorgan positions artificial intelligence and machine learning as important tools to support "
        "risk management, fraud detection, customer personalization, and internal productivity. The strategy is to embed "
        "AI into core processes while keeping strong governance and controls."
    ),
    "what is jpmorgan cloud strategy": (
        "JPMorgan shifts workloads to cloud, adopts hybrid models, strengthens resiliency, and modernizes legacy systems. "
        "Cloud is used to support scalability, reliability, and faster delivery of products."
    ),

    # JPMorgan workforce and investment
    "what is jpmorgan technology workforce": (
        "The document describes a very large technology workforce at JPMorgan, with tens of thousands of engineers "
        "and technologists supporting platforms, infrastructure, cybersecurity, and application development across the firm."
    ),
    "how many engineers does jpmorgan have": (
        "JPMorgan has tens of thousands of engineers and technologists who build and operate its core systems and digital products."
    ),
    "what is jpmorgan technology investment": (
        "The JPMorgan document highlights multi billion yearly technology investment. The bank spends heavily each year "
        "on software development, infrastructure, cybersecurity, data platforms, and digital capabilities to support its businesses."
    ),
    "how much does jpmorgan invest in technology": (
        "The document notes that JPMorgan spends many billions of dollars per year on technology, covering platforms, data, cloud, AI, and infrastructure."
    ),
    "what is jpmorgan technology focus": (
        "JPMorgan technology focus includes cloud, cybersecurity, data platforms, AI, payments modernization, and simplified architecture."
    ),

    # Pfizer general
    "what is pfizer": (
        "Pfizer is a global biopharmaceutical company focused on discovering, developing, and commercializing medicines and vaccines. "
        "The document concentrates on research and development strategy and pipeline."
    ),
    "what is pfizer main motive": (
        "Pfizer main motive in the document is to advance a strong pipeline of innovative medicines and vaccines, "
        "focus on high value therapeutic areas, and turn scientific research into meaningful outcomes for patients while "
        "delivering sustainable growth."
    ),
    "summarise pfizer document": (
        "The Pfizer document describes the company research and development strategy, its pipeline of programs across multiple "
        "therapeutic areas, and the way it prioritizes assets. It explains how Pfizer manages clinical development, regulatory "
        "interactions, and portfolio decisions to bring new medicines and vaccines to market."
    ),

    # Pfizer R and D and pipeline
    "what is pfizer r and d strategy": (
        "Pfizer research and development strategy is to focus on high value areas such as oncology, vaccines, inflammation, "
        "immunology, and rare diseases. The company invests in both internal discovery and external partnerships, uses data and "
        "modern science to select promising targets, and progresses assets through clinical phases with clear go and no go decisions."
    ),
    "what is pfizer pipeline": (
        "The Pfizer document outlines a broad pipeline of development programs across multiple therapeutic areas. "
        "It covers early stage research as well as later stage and near approval assets, and explains how the portfolio is "
        "balanced between risk, scientific novelty, and potential impact on patients."
    ),
    "what challenges does pfizer mention in r and d": (
        "The Pfizer document notes that research and development faces challenges such as scientific uncertainty, "
        "clinical trial complexity, regulatory requirements, competition from other therapies, and the need to manage "
        "resources across many programs. It emphasizes disciplined portfolio management and data driven decision making."
    ),
    "what are pfizer r and d challenges": (
        "Key challenges include scientific risk, clinical trial design, regulatory timelines, and competition from alternative therapies."
    ),
    "what is pfizer pipeline structure": (
        "Pfizer pipeline is structured as a mix of early research projects, mid stage assets, late stage candidates, and near approval or launch programs."
    ),
    "what diseases or areas does pfizer target": (
        "Pfizer focuses on therapeutic areas such as oncology, vaccines, inflammation and immunology, rare diseases, and other high unmet need conditions."
    ),

    # Google general
    "what is google": (
        "Google is a global technology company that operates internet scale products such as search, maps, and YouTube, "
        "and offers cloud and artificial intelligence services through Google Cloud. The document focuses on AI and cloud transformation."
    ),
    "what is google main motive": (
        "Google main motive in the document is to lead in artificial intelligence and cloud services by building advanced models, "
        "scalable infrastructure, and platforms that help customers modernize and innovate."
    ),
    "summarise google document": (
        "The Google document describes how the company is investing in artificial intelligence, cloud infrastructure, and data platforms. "
        "It explains how Google Cloud and AI tools support digital transformation for customers, and how Google is scaling its regions, "
        "infrastructure, and developer ecosystem."
    ),

    # Google AI and cloud
    "what is google ai strategy": (
        "Google AI strategy is to develop state of the art models, integrate AI into its own products, and provide AI platforms "
        "through Google Cloud so that customers can build their own solutions. AI is presented as a core technology that "
        "supports productivity, analytics, and new applications."
    ),
    "what ai technologies does google highlight": (
        "The Google document highlights advanced AI models, large scale training systems, custom accelerators, and enterprise AI solutions delivered through Google Cloud."
    ),
    "what is google cloud strategy": (
        "Google cloud strategy is to provide a secure and scalable cloud platform with strong data, analytics, and AI capabilities. "
        "It focuses on multi cloud and open approaches, industry specific solutions, and close partnership with customers to support "
        "their digital transformation."
    ),
    "what is google cloud transformation strategy": (
        "Google cloud transformation strategy is to help enterprises modernize their applications, adopt data and AI platforms, and run securely at scale across multiple regions."
    ),
    "what is google digital transformation strategy": (
        "Google digital transformation strategy is to help customers modernize using cloud, AI, data platforms, cybersecurity, "
        "and open multi cloud tools that integrate with their existing landscapes."
    ),
    "how is google scaling its ai infrastructure": (
        "The Google document explains that the company is scaling AI infrastructure by expanding data center capacity, "
        "deploying specialized hardware such as custom accelerators, and optimizing its global network. "
        "This supports training and serving large AI models and delivering AI services through the cloud."
    ),
    "what does google say about ai engineers": (
        "The document indicates that Google maintains large engineering and AI teams that work on core models, platforms, "
        "and tools. These engineers focus on improving model quality, reliability, and integration into products and cloud offerings."
    ),
    "what industries is google cloud targeting": (
        "Google Cloud focuses on industries such as financial services, retail, manufacturing, healthcare, media, and the public sector, providing industry specific AI and data solutions."
    ),
    "what are the main themes in the google research": (
        "The main themes in the Google document are AI first strategy, large scale cloud infrastructure, data platforms, industry solutions, and supporting digital transformation for customers."
    ),
}

# Map names mentioned in questions to stock tickers
COMPANY_TICKERS = {
    "jpmorgan": "JPM",
    "jpm": "JPM",
    "pfizer": "PFE",
    "pfe": "PFE",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "googl": "GOOGL",
}


def load_docx(path: str) -> str:
    doc = Document(path)
    parts = []
    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text)
    return "\n".join(parts)


class SimpleIndex:
    def __init__(self):
        # list of {"text": "...", "meta": {"path": "..."}}
        self.docs = []

    def add_doc(self, text, path):
        self.docs.append({"text": text, "meta": {"path": path}})

    def search(self, query, k=6):
        # very simple full text search by substring
        q = query.lower()
        hits = [d for d in self.docs if q in d["text"].lower()]
        if not hits:
            return self.docs[:k]
        return hits[:k]


def build_index():
    idx = SimpleIndex()

    # paths relative to repo root on Streamlit Cloud
    jp_path = "docs/JPMorgan Chase Deep Dive on IT and Digital Transformation Strategy.docx"
    pf_path = "docs/Pfizer – R&D Deep Dive Report.docx"
    gg_path = "docs/Google Deep Dive on AI and Cloud Transformation Strategy.docx"

    for path in [jp_path, pf_path, gg_path]:
        text = load_docx(path)
        idx.add_doc(text, path)

    return idx


def _detect_company_from_query(q: str):
    """
    Try to detect which of the three companies is mentioned.
    Returns (company_name, ticker) or (None, None).
    """
    q_low = q.lower()
    for name, ticker in COMPANY_TICKERS.items():
        if name in q_low:
            return name, ticker
    return None, None


def _get_market_snapshot(ticker: str) -> dict:
    """
    Fetch a simple snapshot of price trend and recent profit using yfinance.
    """
    data = {"price": None, "change_pct": None, "recent_profit": None}

    try:
        stock = yf.Ticker(ticker)

        # recent price trend last 5 days
        hist = stock.history(period="5d")
        if not hist.empty:
            last_price = float(hist["Close"].iloc[-1])
            first_price = float(hist["Close"].iloc[0])
            change_pct = (last_price - first_price) / first_price * 100.0
            data["price"] = round(last_price, 2)
            data["change_pct"] = round(change_pct, 2)

        # try to get most recent annual net income
        try:
            fin = stock.financials
            if fin is not None and not fin.empty and "Net Income" in fin.index:
                recent_profit = float(fin.loc["Net Income"].iloc[0])
                data["recent_profit"] = recent_profit
        except Exception:
            pass

    except Exception:
        # ignore errors and return what we have
        pass

    return data


def _make_market_answer(company_name: str, ticker: str, question: str) -> str:
    """
    Build a natural language answer for market trend, share price, and recent profit.
    """
    snapshot = _get_market_snapshot(ticker)
    parts = []

    price = snapshot.get("price")
    change_pct = snapshot.get("change_pct")
    profit = snapshot.get("recent_profit")

    cname = company_name.title() if company_name else ticker

    if price is not None:
        parts.append(
            f"The current share price of {cname} ticker {ticker} is about {price} in its main market."
        )
        if change_pct is not None:
            direction = "up" if change_pct >= 0 else "down"
            parts.append(
                f"Over the last few trading days the share has moved {direction} by roughly {abs(change_pct)} percent."
            )
    else:
        parts.append(
            f"I could not fetch a reliable current share price for {cname} ticker {ticker} right now."
        )

    if profit is not None:
        parts.append(
            "In the most recent reported year the company shows a positive net income figure, "
            "which you can view as its latest reported profit."
        )
    else:
        parts.append(
            "I could not read a clear recent profit figure from the public financial data in this environment."
        )

    parts.append(
        "These numbers come from live market data, not from the uploaded research documents."
    )

    return " ".join(parts)


def _get_company_context(index, company_name: str | None, query: str, detail: bool) -> str:
    """
    If a company is detected in the question, return the full text of that company document.
    Otherwise use the simple search over all docs.
    """
    texts = []

    if company_name:
        cname = company_name.lower()
        for d in index.docs:
            path = d["meta"].get("path", "").lower()
            if cname in path:
                texts.append(d["text"])

    # if we did not find any company specific doc, fall back to search
    if not texts:
        k = 6 if detail else 3
        hits = index.search(query, k=k)
        texts = [h["text"] for h in hits]

    context = " ".join(_norm(t) for t in texts)

    # limit context size to control tokens
    max_chars = 4000
    if len(context) > max_chars:
        context = context[:max_chars]

    return context


def _scan_pfizer_facts(index):
    """
    Scan Pfizer document once to extract pipeline, phase 3 and under review counts.
    Rule based, no LLM needed.
    """
    pf_text = ""
    for d in index.docs:
        path = d["meta"].get("path", "").lower()
        if "pfizer" in path:
            pf_text += " " + d["text"]

    pf_text = _norm(pf_text)

    facts = {"pipeline": None, "phase3": None, "under_review": None}

    m = re.search(r"(\d{2,3})\s+(development\s+)?programs", pf_text, re.I)
    if m:
        facts["pipeline"] = m.group(1)

    m = re.search(r"(\d{1,3})\s+candidates?\s+in\s+phase\s*3", pf_text, re.I)
    if m:
        facts["phase3"] = m.group(1)

    m = re.search(r"(\d{1,3})\s+under regulatory review", pf_text, re.I)
    if m:
        facts["under_review"] = m.group(1)

    return facts


def _answer_pfizer_programs(index, question: str) -> str:
    """
    Direct answer for questions like:
    - How many programs does Pfizer have
    - What is Pfizer pipeline size
    Works even if the LLM is rate limited.
    """
    facts = _scan_pfizer_facts(index)
    parts = []

    if facts.get("pipeline"):
        parts.append(f"Based on the Pfizer document the research and development pipeline includes about {facts['pipeline']} programs.")
    if facts.get("phase3"):
        parts.append(f"There are around {facts['phase3']} candidates in phase 3 development.")
    if facts.get("under_review"):
        parts.append(f"Roughly {facts['under_review']} programs are described as under regulatory review.")

    if not parts:
        return "I could not read a clear number of programs for Pfizer from the document."

    return " ".join(parts)


def _make_llm_answer(question: str, context: str, detail: bool = False) -> str:
    """
    Ask the OpenAI model to answer using only the provided context.
    Uses a simple cache to reduce repeated calls.
    """
    if not context.strip():
        return "I could not find relevant content in the documents for this question."

    # simple cache key uses question and detail flag only
    cache_key = (question.strip().lower(), detail)
    if cache_key in LLM_CACHE:
        return LLM_CACHE[cache_key]

    system_content = (
        "You are a careful research assistant. Answer the question using only the context below. "
        "Do not invent facts. If the context does not clearly contain the answer, say that you do not see it clearly."
    )

    if detail:
        system_content += " Give a detailed explanation with several clear paragraphs."
    else:
        system_content += " Give a concise answer in one or two paragraphs that still explains the main idea."

    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": f"Context from documents:\n{context}\n\nQuestion: {question}",
        },
    ]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=300,
        )
        answer = completion.choices[0].message.content.strip()
        LLM_CACHE[cache_key] = answer
        return answer

    except RateLimitError:
        return (
            "I am calling the language model too often and have reached the current rate limit. "
            "Please try again after a short time."
        )

    except Exception as e:
        return f"An error occurred while generating the answer: {e}"


def smart_answer(index, query: str, detail: bool = False) -> str:
    """
    Main entry point used by the Streamlit app.

    1. If the question is a greeting or meta question, answer from FAQ.
    2. If the question asks about market trend, share price, performance, or recent profit,
       answer using live market data for the three companies.
    3. If the question is about Pfizer program count or pipeline size, answer with rule based facts.
    4. Otherwise, detect which company is mentioned and feed that document
       to the LLM to answer from context.
    """
    nq = _norm_question(query)

    # 1. FAQ for exact matches only
    if nq in FAQ_QA:
        return f"Question  {query}\n\nAnswer\n{FAQ_QA[nq]}"

    q_low = query.lower()

    # 2. Market related questions for the three companies
    market_keywords = [
        "market trend",
        "share price",
        "stock price",
        "nifty share",
        "recent profit",
        "latest profit",
        "earnings",
        "recent earnings",
        "recent performance",
        "stock performance",
        "price performance",
    ]
    is_market_question = any(kw in q_low for kw in market_keywords)

    if is_market_question:
        company_name, ticker = _detect_company_from_query(query)
        # treat questions like "google recent performance" as market even without explicit ticker
        if not ticker and "google" in q_low:
            company_name, ticker = "google", "GOOGL"
        if not ticker and "pfizer" in q_low:
            company_name, ticker = "pfizer", "PFE"
        if not ticker and ("jpmorgan" in q_low or "jpm" in q_low):
            company_name, ticker = "jpmorgan", "JPM"

        if ticker:
            market_text = _make_market_answer(company_name, ticker, query)
            return f"Question  {query}\n\nAnswer\n{market_text}"

    # 3. Pfizer program and pipeline questions answered rule based, without LLM
    if "pfizer" in q_low and any(w in q_low for w in ["program", "pipeline", "phase 3", "phase three"]):
        pf_answer = _answer_pfizer_programs(index, query)
        return f"Question  {query}\n\nAnswer\n{pf_answer}"

    # 4. Normal document based answer through the LLM
    company_name, _ = _detect_company_from_query(query)

    context = _get_company_context(index, company_name, query, detail)

    answer_text = _make_llm_answer(query, context, detail=detail)

    return f"Question  {query}\n\nAnswer\n{answer_text}"
