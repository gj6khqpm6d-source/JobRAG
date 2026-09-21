<img src="https://github.com/cullenwatson/JobSpy/assets/78247585/ae185b7e-e444-4712-8bb9-fa97f53e896b" width="400">

**JobSpy** is a job scraping library with the goal of aggregating all the jobs from popular job boards with one tool.

## Local Web App

JobRAG 的业务目标、阶段划分、验收标准和生产化要求记录在
[JobRAG 业务设计骨架](docs/jobrag_business_design.md) 中。后续开发以该文档为指导，具体实现阶段需先确认设计，再执行和验收。

This repository includes a local web interface for all JobSpy sites and search options.

```bash
chmod +x run.sh
./run.sh
```

Open <http://127.0.0.1:8000>. The first run creates an isolated Python 3.11 environment and installs dependencies. Set a different port with `JOBSPY_PORT=9000 ./run.sh`.

The web app runs searches in the background, keeps partial results if an individual board is blocked, and exports the current result set to CSV or Excel. Search results live in memory and are cleared when the server restarts.

Every successful scrape is also persisted to the JobRAG knowledge base. Without configuration the app uses `data/jobrag.db` as a development fallback. The target deployment uses PostgreSQL with pgvector:

```bash
cp .env.example .env
docker compose up -d postgres
# Set DATABASE_URL in .env to the PostgreSQL example, then:
.venv/bin/alembic upgrade head
./run.sh
```

Knowledge-base APIs:

- `GET /api/kb/stats` — stored job and snapshot counts
- `GET /api/kb/contract` — active machine-readable JobRAG business contract
- `GET /api/kb/jobs` — browse persisted jobs with `query`, `source`, `limit`, and `offset`
- `POST /api/kb/backfill/linkedin` — backfill missing LinkedIn descriptions and create chunks
- `GET /api/kb/index` — view chunk and embedding index status
- `POST /api/kb/index` — index pending chunks with local BGE-M3
- `POST /api/kb/retrieve` — hybrid retrieval with optional metadata filters
- `POST /api/kb/ask` — evidence-grounded answer generation with DeepSeek

Embeddings run locally with `BAAI/bge-m3` and require no API key. The model is downloaded into `data/models` on first use and reused offline afterwards. Cloudflare remains available as an optional fallback. Keep any API credentials in `.env`; never commit that file.

LinkedIn full-description retrieval is enabled by default because jobs without descriptions cannot participate in RAG. The knowledge-base page reports total jobs, searchable jobs, missing descriptions, and description coverage separately. Existing LinkedIn rows can be repaired with **补全缺失描述** before generating their pending local vectors.

### Enable RAG

1. Run `./run.sh`, open **RAG 知识库**, and click **生成本地向量**. The first run downloads BGE-M3; later runs use the local cache.
2. When answer generation is needed, add `DEEPSEEK_API_KEY` and choose a `DEEPSEEK_MODEL` in `.env`, then restart again.

The current development mode uses SQLite so scraping, persistence, chunking, and the UI work without Docker. For the target PostgreSQL deployment, install Docker Desktop (or PostgreSQL with pgvector), switch `DATABASE_URL`, run `alembic upgrade head`, and restart the app. The PostgreSQL migration creates an HNSW vector index and a GIN full-text index; retrieval fuses vector and keyword rankings and keeps metadata filters in the database query.

SQLite retrieval now creates a local FTS5/BM25 index (`job_chunks_fts`) when chunks are prepared. The index is fused with BGE-M3 results by rank (RRF), so exact terms such as model names and technologies complement semantic matches. The query layer also adds a small, deterministic Chinese-to-English retrieval vocabulary (for example `实习` → `internship`, `岗位要求` → `requirements`) while preserving the original question for answer generation. If the Python SQLite build does not include FTS5, retrieval falls back to the deterministic lexical scorer.

After adding or changing jobs, use `POST /api/kb/index` (or **生成本地向量** in the UI) to fill pending embeddings. FTS5 is rebuilt automatically when chunks are prepared; it does not require a separate service.

RAG data flow:

```text
JobSpy scrape -> normalize/deduplicate -> job snapshots -> semantic chunks
             -> local BGE-M3 embeddings -> pgvector + full-text retrieval
             -> evidence-only DeepSeek prompt -> answer with job citations
```

岗位描述进入知识库前会按业务相关章节进行结构化处理：保留职责、任职要求、技能、经验、教育和项目等章节，过滤公司背景、福利、薪资、申请流程和法律声明；章节内优先按段落或列表项成块，只有超长文本才使用重叠切分。

问答请求会先经过确定性路由：具体岗位要求走证据检索；统计类问题走岗位分析流程；明显不属于招聘知识库的问题直接拒答。当前开发实现仍保留部分全库统计逻辑作为过渡，目标业务行为是先筛选相关岗位候选集，再进行岗位级技能统计，详见 [JobRAG 业务设计骨架](docs/jobrag_business_design.md)。检索结果和最终答案都写入本地、带语料版本号的 SQLite 缓存；新岗位、片段变化或向量回填后版本自动变化，旧缓存不会被复用。

## Features

- Scrapes job postings from **LinkedIn**, **Indeed**, **Glassdoor**, **Google**, **ZipRecruiter**, & other job boards concurrently
- Aggregates the job postings in a dataframe
- Proxies support to bypass blocking

![jobspy](https://github.com/cullenwatson/JobSpy/assets/78247585/ec7ef355-05f6-4fd3-8161-a817e31c5c57)

### Installation

```
pip install -U python-jobspy
```

_Python version >= [3.10](https://www.python.org/downloads/release/python-3100/) required_

### Usage

```python
import csv
from jobspy import scrape_jobs

jobs = scrape_jobs(
    site_name=["indeed", "linkedin", "zip_recruiter", "google"], # "glassdoor", "bayt", "naukri", "bdjobs"
    search_term="software engineer",
    google_search_term="software engineer jobs near San Francisco, CA since yesterday",
    location="San Francisco, CA",
    results_wanted=20,
    hours_old=72,
    country_indeed='USA',
    
    # linkedin_fetch_description=True # gets more info such as description, direct job url (slower)
    # proxies=["208.195.175.46:65095", "208.195.175.45:65095", "localhost"],
)
print(f"Found {len(jobs)} jobs")
print(jobs.head())
jobs.to_csv("jobs.csv", quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False) # to_excel
```

### Output

```
SITE           TITLE                             COMPANY           CITY          STATE  JOB_TYPE  INTERVAL  MIN_AMOUNT  MAX_AMOUNT  JOB_URL                                            DESCRIPTION
indeed         Software Engineer                 AMERICAN SYSTEMS  Arlington     VA     None      yearly    200000      150000      https://www.indeed.com/viewjob?jk=5e409e577046...  THIS POSITION COMES WITH A 10K SIGNING BONUS!...
indeed         Senior Software Engineer          TherapyNotes.com  Philadelphia  PA     fulltime  yearly    135000      110000      https://www.indeed.com/viewjob?jk=da39574a40cb...  About Us TherapyNotes is the national leader i...
linkedin       Software Engineer - Early Career  Lockheed Martin   Sunnyvale     CA     fulltime  yearly    None        None        https://www.linkedin.com/jobs/view/3693012711      Description:By bringing together people that u...
linkedin       Full-Stack Software Engineer      Rain              New York      NY     fulltime  yearly    None        None        https://www.linkedin.com/jobs/view/3696158877      Rain’s mission is to create the fastest and ea...
zip_recruiter Software Engineer - New Grad       ZipRecruiter      Santa Monica  CA     fulltime  yearly    130000      150000      https://www.ziprecruiter.com/jobs/ziprecruiter...  We offer a hybrid work environment. Most US-ba...
zip_recruiter Software Developer                 TEKsystems        Phoenix       AZ     fulltime  hourly    65          75          https://www.ziprecruiter.com/jobs/teksystems-0...  Top Skills' Details• 6 years of Java developme...

```

### Parameters for `scrape_jobs()`

```plaintext
Optional
├── site_name (list|str): 
|    linkedin, zip_recruiter, indeed, glassdoor, google, bayt, bdjobs
|    (default is all)
│
├── search_term (str)
|
├── google_search_term (str)
|     search term for google jobs. This is the only param for filtering google jobs.
│
├── location (str)
│
├── distance (int): 
|    in miles, default 50
│
├── job_type (str): 
|    fulltime, parttime, internship, contract
│
├── proxies (list): 
|    in format ['user:pass@host:port', 'localhost']
|    each job board scraper will round robin through the proxies
|
├── is_remote (bool)
│
├── results_wanted (int): 
|    number of job results to retrieve for each site specified in 'site_name'
│
├── easy_apply (bool): 
|    filters for jobs that are hosted on the job board site (LinkedIn easy apply filter no longer works)
|
├── user_agent (str): 
|    override the default user agent which may be outdated
│
├── description_format (str): 
|    markdown, html (Format type of the job descriptions. Default is markdown.)
│
├── offset (int): 
|    starts the search from an offset (e.g. 25 will start the search from the 25th result)
│
├── hours_old (int): 
|    filters jobs by the number of hours since the job was posted 
|    (ZipRecruiter and Glassdoor round up to next day.)
│
├── verbose (int) {0, 1, 2}: 
|    Controls the verbosity of the runtime printouts 
|    (0 prints only errors, 1 is errors+warnings, 2 is all logs. Default is 2.)

├── linkedin_fetch_description (bool): 
|    fetches full description and direct job url for LinkedIn (Increases requests by O(n))
│
├── linkedin_company_ids (list[int]): 
|    searches for linkedin jobs with specific company ids
|
├── country_indeed (str): 
|    filters the country on Indeed & Glassdoor (see below for correct spelling)
|
├── enforce_annual_salary (bool): 
|    converts wages to annual salary
|
├── ca_cert (str)
|    path to CA Certificate file for proxies
```

```
├── Indeed limitations:
|    Only one from this list can be used in a search:
|    - hours_old
|    - job_type & is_remote
|    - easy_apply
│
└── LinkedIn limitations:
|    Only one from this list can be used in a search:
|    - hours_old
|    - easy_apply
```

## Supported Countries for Job Searching

### **LinkedIn**

LinkedIn searches globally & uses only the `location` parameter. 

### **ZipRecruiter**

ZipRecruiter searches for jobs in **US/Canada** & uses only the `location` parameter.

### **Indeed / Glassdoor**

Indeed & Glassdoor supports most countries, but the `country_indeed` parameter is required. Additionally, use the `location`
parameter to narrow down the location, e.g. city & state if necessary. 

You can specify the following countries when searching on Indeed (use the exact name, * indicates support for Glassdoor):

|                      |              |            |                |
|----------------------|--------------|------------|----------------|
| Argentina            | Australia*   | Austria*   | Bahrain        |
| Belgium*             | Brazil*      | Canada*    | Chile          |
| China                | Colombia     | Costa Rica | Czech Republic |
| Denmark              | Ecuador      | Egypt      | Finland        |
| France*              | Germany*     | Greece     | Hong Kong*     |
| Hungary              | India*       | Indonesia  | Ireland*       |
| Israel               | Italy*       | Japan      | Kuwait         |
| Luxembourg           | Malaysia     | Mexico*    | Morocco        |
| Netherlands*         | New Zealand* | Nigeria    | Norway         |
| Oman                 | Pakistan     | Panama     | Peru           |
| Philippines          | Poland       | Portugal   | Qatar          |
| Romania              | Saudi Arabia | Singapore* | South Africa   |
| South Korea          | Spain*       | Sweden     | Switzerland*   |
| Taiwan               | Thailand     | Turkey     | Ukraine        |
| United Arab Emirates | UK*          | USA*       | Uruguay        |
| Venezuela            | Vietnam*     |            |                |

### **Bayt**

Bayt only uses the search_term parameter currently and searches internationally



## Notes
* Indeed is the best scraper currently with no rate limiting.  
* All the job board endpoints are capped at around 1000 jobs on a given search.  
* LinkedIn is the most restrictive and usually rate limits around the 10th page with one ip. Proxies are a must basically.

## Frequently Asked Questions

---
**Q: Why is Indeed giving unrelated roles?**  
**A:** Indeed searches the description too.

- use - to remove words
- "" for exact match

Example of a good Indeed query

```py
search_term='"engineering intern" software summer (java OR python OR c++) 2025 -tax -marketing'
```

This searches the description/title and must include software, summer, 2025, one of the languages, engineering intern exactly, no tax, no marketing.

---

**Q: No results when using "google"?**  
**A:** You have to use super specific syntax. Search for google jobs on your browser and then whatever pops up in the google jobs search box after applying some filters is what you need to copy & paste into the google_search_term. 

---

**Q: Received a response code 429?**  
**A:** This indicates that you have been blocked by the job board site for sending too many requests. All of the job board sites are aggressive with blocking. We recommend:

- Wait some time between scrapes (site-dependent).
- Try using the proxies param to change your IP address.

---

### JobPost Schema

```plaintext
JobPost
├── title
├── company
├── company_url
├── job_url
├── location
│   ├── country
│   ├── city
│   ├── state
├── is_remote
├── description
├── job_type: fulltime, parttime, internship, contract
├── job_function
│   ├── interval: yearly, monthly, weekly, daily, hourly
│   ├── min_amount
│   ├── max_amount
│   ├── currency
│   └── salary_source: direct_data, description (parsed from posting)
├── date_posted
└── emails

Linkedin specific
└── job_level

Linkedin & Indeed specific
└── company_industry

Indeed specific
├── company_country
├── company_addresses
├── company_employees_label
├── company_revenue_label
├── company_description
└── company_logo

Naukri specific
├── skills
├── experience_range
├── company_rating
├── company_reviews_count
├── vacancy_count
└── work_from_home_type
```
