from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RoleProfile:
    role: str
    industry: str
    career_category: str
    sub_domain: str
    keywords: tuple[str, ...]
    skills: tuple[str, ...]
    improvements: tuple[str, ...]
    certifications: tuple[str, ...]
    roadmap: tuple[str, ...]
    metric_examples: tuple[str, ...]
    portfolio_terms: tuple[str, ...] = ()
    avoid_terms: tuple[str, ...] = ()


TECH_AVOID_FOR_NON_TECH = (
    "react",
    "node",
    "java",
    "javascript",
    "python",
    "docker",
    "kubernetes",
    "api",
    "backend",
    "frontend",
    "dsa",
    "github",
    "git",
    "leetcode",
)


ROLE_PROFILES: tuple[RoleProfile, ...] = (
    RoleProfile(
        role="Frontend Developer",
        industry="Software & IT",
        career_category="Technology",
        sub_domain="Web UI Engineering",
        keywords=(
            "frontend",
            "front end",
            "react",
            "next.js",
            "javascript",
            "typescript",
            "html",
            "css",
            "tailwind",
            "responsive",
            "web app",
            "ui",
            "accessibility",
        ),
        skills=("react", "javascript", "typescript", "html", "css", "tailwind", "accessibility", "testing", "git"),
        improvements=(
            "Add live links for polished UI work and mention measurable usability or performance outcomes.",
            "Group frontend skills by framework, styling, testing, state management, and accessibility.",
            "Rewrite project bullets with user impact, performance gains, conversion changes, or responsiveness results.",
        ),
        certifications=("Meta Front-End Developer", "Google UX Design", "freeCodeCamp Responsive Web Design"),
        roadmap=(
            "Audit two projects for responsive behavior, accessibility, and measurable UI outcomes.",
            "Strengthen React patterns, state management, testing, and component architecture.",
            "Build or refine one production-quality interface with performance and accessibility notes.",
            "Tailor the resume to frontend roles with role keywords, live links, and quantified project bullets.",
        ),
        metric_examples=("Improved page load speed by 35%", "Reduced checkout drop-off by 18%", "Raised Lighthouse score to 95+"),
    ),
    RoleProfile(
        role="Backend Developer",
        industry="Software & IT",
        career_category="Technology",
        sub_domain="API & Server Engineering",
        keywords=("backend", "back end", "api", "rest", "graphql", "database", "server", "microservices", "django", "flask", "spring", "node"),
        skills=("api design", "rest api", "sql", "postgresql", "mongodb", "authentication", "docker", "aws", "testing", "system design"),
        improvements=(
            "Add API scale, latency, reliability, security, and database outcomes to backend bullets.",
            "Separate backend skills into languages, frameworks, databases, cloud, testing, and observability.",
            "Include links or concise descriptions for services that demonstrate authentication, persistence, and deployment.",
        ),
        certifications=("AWS Certified Cloud Practitioner", "PostgreSQL Associate", "Docker Foundations"),
        roadmap=(
            "Map existing projects to API, database, authentication, and deployment evidence.",
            "Strengthen one backend service with tests, logging, and database design notes.",
            "Add cloud or container deployment proof and reliability metrics.",
            "Tailor resume bullets to backend ownership, scale, security, and maintainability.",
        ),
        metric_examples=("Reduced API latency by 40%", "Handled 10k+ requests per day", "Improved deployment reliability by 30%"),
    ),
    RoleProfile(
        role="Full Stack Developer",
        industry="Software & IT",
        career_category="Technology",
        sub_domain="End-to-End Web Applications",
        keywords=("full stack", "mern", "frontend", "backend", "react", "node", "django", "flask", "database", "api", "deployment"),
        skills=("react", "javascript", "api design", "sql", "mongodb", "authentication", "testing", "docker", "git", "cloud deployment"),
        improvements=(
            "Show end-to-end ownership: UI, API, database, deployment, and measurable product result.",
            "Make project bullets clear about the problem, architecture, users, and measurable improvement.",
            "Add repository and live links when the work is public and production-ready.",
        ),
        certifications=("AWS Certified Cloud Practitioner", "MongoDB Associate Developer", "Meta Front-End Developer"),
        roadmap=(
            "Choose one strongest full-stack project and document its architecture clearly.",
            "Improve API, auth, testing, and deployment evidence.",
            "Add measurable product outcomes and screenshots or links.",
            "Tailor each application around the stack named in the target job.",
        ),
        metric_examples=("Built end-to-end app used by 500+ users", "Reduced manual processing by 45%", "Improved release cycle time by 30%"),
    ),
    RoleProfile(
        role="Software Engineer",
        industry="Software & IT",
        career_category="Technology",
        sub_domain="Software Development",
        keywords=("software engineer", "developer", "programming", "application", "algorithm", "system", "debugging", "testing", "deployment"),
        skills=("programming", "data structures", "testing", "system design", "databases", "git", "debugging", "cloud", "documentation"),
        improvements=(
            "Show engineering ownership with problem, technical approach, collaboration, and measurable outcome.",
            "Group skills by language, framework, database, tooling, and cloud instead of one long list.",
            "Add links to strong engineering projects only when the code is clean and representative.",
        ),
        certifications=("AWS Certified Cloud Practitioner", "Microsoft Azure Fundamentals", "Google Associate Cloud Engineer"),
        roadmap=(
            "Clarify strongest engineering projects and the problems they solved.",
            "Add testing, architecture, deployment, and collaboration evidence.",
            "Strengthen missing fundamentals for target roles.",
            "Tailor resume keywords and measurable outcomes for each application.",
        ),
        metric_examples=("Reduced processing time by 40%", "Automated workflow saving 8 hours weekly", "Improved defect resolution time by 25%"),
    ),
    RoleProfile(
        role="Data Analyst",
        industry="Data & Analytics",
        career_category="Technology",
        sub_domain="Business Intelligence",
        keywords=("data analyst", "analytics", "dashboard", "sql", "excel", "power bi", "tableau", "reporting", "insights", "kpi"),
        skills=("sql", "excel", "power bi", "tableau", "python", "pandas", "data cleaning", "dashboarding", "statistics"),
        improvements=(
            "Add business metrics, dashboard users, decision impact, and reporting cadence.",
            "Show data cleaning, SQL querying, visualization, and stakeholder communication separately.",
            "Use outcome language around revenue, cost, efficiency, forecast accuracy, or KPI visibility.",
        ),
        certifications=("Microsoft Power BI Data Analyst", "Google Data Analytics", "Tableau Desktop Specialist"),
        roadmap=(
            "Audit resume for SQL, dashboard, KPI, and business impact evidence.",
            "Build one dashboard case study with clear problem, data, insights, and decision outcome.",
            "Strengthen statistics and stakeholder storytelling.",
            "Tailor keywords to analyst postings and quantify reporting impact.",
        ),
        metric_examples=("Reduced reporting time by 50%", "Improved forecast accuracy by 12%", "Built dashboard used by 20+ stakeholders"),
    ),
    RoleProfile(
        role="UI/UX Designer",
        industry="Design",
        career_category="Creative & Product",
        sub_domain="Product Design",
        keywords=("ui/ux", "ux", "user experience", "user interface", "figma", "wireframe", "prototype", "usability", "case study", "behance"),
        skills=("figma", "wireframing", "prototyping", "user research", "usability testing", "information architecture", "design systems"),
        improvements=(
            "Add portfolio, Behance, or case study links that show process, constraints, and results.",
            "Describe design impact using usability findings, conversion, task success, accessibility, or adoption metrics.",
            "Separate research, interaction design, visual design, prototyping, and handoff skills.",
        ),
        certifications=("Google UX Design", "Nielsen Norman Group UX Certification", "Human-Computer Interaction certificate"),
        roadmap=(
            "Turn one project into a concise case study with problem, users, process, and outcome.",
            "Strengthen Figma components, design systems, accessibility, and handoff evidence.",
            "Add usability testing or research artifacts to portfolio.",
            "Tailor resume to product design roles with portfolio links and outcome metrics.",
        ),
        metric_examples=("Improved task completion by 28%", "Reduced onboarding friction by 20%", "Validated design with 12 user interviews"),
        portfolio_terms=("portfolio", "behance", "figma", "dribbble", "case study"),
        avoid_terms=("dsa", "backend roadmap", "docker", "kubernetes", "server architecture"),
    ),
    RoleProfile(
        role="Doctor",
        industry="Healthcare",
        career_category="Clinical",
        sub_domain="Medical Practice",
        keywords=("doctor", "physician", "medical officer", "mbbs", "md", "clinic", "hospital", "patient", "diagnosis", "treatment", "rounds", "opd", "icu"),
        skills=("patient care", "diagnosis", "clinical documentation", "treatment planning", "emergency care", "case management", "medical ethics"),
        improvements=(
            "Add clinical specialization, patient volume, procedures, rotations, and hospital experience.",
            "Quantify patient outcomes, caseload, emergency exposure, audits, or quality improvements where truthful.",
            "Highlight licenses, registrations, certifications, publications, conferences, and clinical training.",
        ),
        certifications=("BLS", "ACLS", "Specialty fellowship", "Medical council registration", "Clinical research certification"),
        roadmap=(
            "Clarify clinical specialty, rotations, patient populations, and hospital settings.",
            "Add certifications, registration details, procedures, and case exposure.",
            "Document patient-care outcomes, audits, research, or quality initiatives.",
            "Tailor resume for the target hospital, department, or specialization.",
        ),
        metric_examples=("Managed 30+ patients per shift", "Assisted 100+ procedures", "Reduced discharge documentation delays by 25%"),
        avoid_terms=TECH_AVOID_FOR_NON_TECH,
    ),
    RoleProfile(
        role="Nurse / Healthcare Professional",
        industry="Healthcare",
        career_category="Clinical",
        sub_domain="Patient Care",
        keywords=("nurse", "nursing", "healthcare", "patient care", "ward", "icu", "vital signs", "medication", "clinical", "hospital"),
        skills=("patient care", "medication administration", "vital signs", "infection control", "emergency response", "care planning"),
        improvements=(
            "Add patient-care settings, ward exposure, shift responsibilities, and clinical procedures.",
            "Quantify caseload, patient satisfaction, safety outcomes, or process improvements where truthful.",
            "Highlight licenses, BLS/ACLS, infection control, and specialty unit experience.",
        ),
        certifications=("BLS", "ACLS", "Infection Control", "Critical Care Nursing", "Nursing council registration"),
        roadmap=(
            "Clarify clinical setting, patient population, and responsibilities.",
            "Add license and relevant patient-care certifications.",
            "Quantify caseload, safety practices, and outcomes.",
            "Tailor the resume to the unit or healthcare setting being targeted.",
        ),
        metric_examples=("Cared for 20+ patients per shift", "Maintained 98% medication documentation accuracy", "Supported 15+ emergency responses"),
        avoid_terms=TECH_AVOID_FOR_NON_TECH,
    ),
    RoleProfile(
        role="Teacher",
        industry="Education",
        career_category="Education",
        sub_domain="Teaching & Learning",
        keywords=("teacher", "teaching", "classroom", "curriculum", "lesson plan", "students", "school", "education", "pedagogy", "assessment"),
        skills=("lesson planning", "classroom management", "curriculum design", "student assessment", "parent communication", "differentiated instruction"),
        improvements=(
            "Add classroom impact, student improvement, teaching methods, lesson design, and assessment outcomes.",
            "Quantify class size, pass-rate improvement, learning gains, events led, or parent engagement.",
            "Highlight certifications, subject expertise, curriculum work, and student-support initiatives.",
        ),
        certifications=("B.Ed", "Teaching license", "TESOL/TEFL", "Subject teaching certification", "Child psychology training"),
        roadmap=(
            "Clarify subject, grade level, curriculum, and classroom responsibilities.",
            "Add student outcomes, assessment data, and teaching methods.",
            "Highlight certifications, workshops, and parent/student engagement.",
            "Tailor resume to school type, subject, and grade level.",
        ),
        metric_examples=("Improved class average by 18%", "Taught 120+ students annually", "Raised assignment completion by 30%"),
        avoid_terms=TECH_AVOID_FOR_NON_TECH,
    ),
    RoleProfile(
        role="Human Resources",
        industry="Human Resources",
        career_category="People Operations",
        sub_domain="Recruitment & Employee Experience",
        keywords=("human resources", "hr", "recruitment", "talent acquisition", "onboarding", "employee engagement", "payroll", "hrms", "interview"),
        skills=("recruitment", "sourcing", "interview coordination", "onboarding", "employee engagement", "hr policies", "communication", "hrms"),
        improvements=(
            "Add hiring metrics, time-to-fill, offer acceptance, onboarding speed, and employee engagement outcomes.",
            "Show communication, stakeholder handling, policy work, HRMS tools, and recruitment funnel ownership.",
            "Use people-operations language instead of generic administration bullets.",
        ),
        certifications=("SHRM-CP", "HRCI aPHR/PHR", "HR Analytics", "Talent Acquisition certification"),
        roadmap=(
            "Clarify HR function: recruitment, operations, engagement, payroll, or generalist work.",
            "Add funnel metrics, hiring volume, onboarding outcomes, and stakeholder evidence.",
            "Strengthen HRMS, labor law, analytics, or employee relations keywords.",
            "Tailor resume to HR generalist or talent acquisition postings.",
        ),
        metric_examples=("Closed 25+ roles in a quarter", "Reduced time-to-fill by 22%", "Improved onboarding completion to 95%"),
        avoid_terms=TECH_AVOID_FOR_NON_TECH,
    ),
    RoleProfile(
        role="Marketing Professional",
        industry="Marketing",
        career_category="Growth & Communication",
        sub_domain="Digital Marketing",
        keywords=("marketing", "campaign", "seo", "sem", "social media", "content", "brand", "lead generation", "google ads", "analytics"),
        skills=("campaign management", "seo", "content strategy", "social media", "google ads", "analytics", "copywriting", "brand communication"),
        improvements=(
            "Add campaign outcomes such as leads, conversion, reach, engagement, revenue, or CAC changes.",
            "Separate channels: SEO, paid media, content, email, social, analytics, and brand.",
            "Include portfolio links or campaign samples when they show strategic and measurable work.",
        ),
        certifications=("Google Ads", "Google Analytics", "HubSpot Content Marketing", "Meta Blueprint"),
        roadmap=(
            "Audit resume for campaign metrics and channel ownership.",
            "Build one campaign case study with target audience, channel, budget, and result.",
            "Strengthen analytics, attribution, SEO, or paid media evidence.",
            "Tailor resume to growth, content, brand, or performance marketing roles.",
        ),
        metric_examples=("Increased qualified leads by 35%", "Improved email CTR by 18%", "Reduced CAC by 12%"),
        portfolio_terms=("portfolio", "campaign", "case study", "website"),
        avoid_terms=("dsa", "backend roadmap", "docker", "kubernetes"),
    ),
    RoleProfile(
        role="Finance Professional",
        industry="Finance",
        career_category="Business & Finance",
        sub_domain="Accounting / Financial Analysis",
        keywords=("finance", "accounting", "financial analysis", "budget", "audit", "tax", "forecast", "valuation", "ledger", "excel", "sap"),
        skills=("financial analysis", "excel", "budgeting", "forecasting", "accounting", "audit", "taxation", "sap", "financial reporting"),
        improvements=(
            "Add financial impact: cost savings, reporting accuracy, budget size, variance reduction, or audit outcomes.",
            "Separate accounting, reporting, analysis, tax, audit, ERP, and spreadsheet skills.",
            "Highlight certifications, compliance exposure, and stakeholder reporting.",
        ),
        certifications=("CPA", "CFA", "ACCA", "Financial Modeling & Valuation", "Tally / SAP certification"),
        roadmap=(
            "Clarify finance function: accounting, analysis, audit, taxation, or FP&A.",
            "Add numbers for budget size, cost savings, reporting speed, or accuracy.",
            "Strengthen Excel, financial modeling, ERP, and compliance evidence.",
            "Tailor resume to finance analyst, accountant, audit, or FP&A roles.",
        ),
        metric_examples=("Reduced monthly close time by 30%", "Managed INR 2 crore budget tracking", "Improved forecast accuracy by 15%"),
        avoid_terms=TECH_AVOID_FOR_NON_TECH,
    ),
    RoleProfile(
        role="Mechanical Engineer",
        industry="Engineering",
        career_category="Engineering",
        sub_domain="Mechanical Design / Manufacturing",
        keywords=("mechanical", "solidworks", "autocad", "cad", "manufacturing", "production", "maintenance", "thermal", "cnc", "quality"),
        skills=("solidworks", "autocad", "cad", "manufacturing", "gd&t", "quality control", "maintenance", "cnc", "lean manufacturing"),
        improvements=(
            "Add CAD tools, manufacturing exposure, design calculations, production metrics, and quality outcomes.",
            "Quantify cycle-time reduction, defect reduction, cost savings, equipment uptime, or design improvements.",
            "Highlight internships, shop-floor exposure, standards, and mechanical project documentation.",
        ),
        certifications=("SolidWorks CSWA/CSWP", "AutoCAD", "Lean Six Sigma", "GD&T", "NDT certification"),
        roadmap=(
            "Clarify target path: design, manufacturing, maintenance, quality, or production.",
            "Add CAD models, calculations, process improvements, and measurable plant/project outcomes.",
            "Strengthen standards, GD&T, quality tools, and manufacturing methods.",
            "Tailor resume to mechanical design or manufacturing job descriptions.",
        ),
        metric_examples=("Reduced cycle time by 18%", "Improved machine uptime by 12%", "Lowered material waste by 10%"),
        avoid_terms=("react", "frontend roadmap", "backend", "api", "dsa", "github"),
    ),
    RoleProfile(
        role="Civil Engineer",
        industry="Engineering",
        career_category="Engineering",
        sub_domain="Construction / Structural",
        keywords=("civil", "construction", "site engineer", "structural", "estimation", "quantity surveying", "autocad", "staad", "revit", "boq"),
        skills=("autocad", "revit", "staad pro", "estimation", "quantity surveying", "site supervision", "project planning", "quality control"),
        improvements=(
            "Add site/project details, quantities, budgets, safety practices, drawings, and construction outcomes.",
            "Quantify project size, cost control, schedule adherence, quality checks, or material savings.",
            "Highlight software, standards, site supervision, vendor coordination, and documentation.",
        ),
        certifications=("AutoCAD", "Revit", "STAAD.Pro", "Primavera P6", "Construction Safety"),
        roadmap=(
            "Clarify civil path: site, structural, planning, estimation, or QA/QC.",
            "Add project scale, drawings, BOQ, safety, and site coordination evidence.",
            "Strengthen AutoCAD/Revit/STAAD/Primavera skills as relevant.",
            "Tailor resume to construction, structural, or planning roles.",
        ),
        metric_examples=("Monitored 50,000 sq ft site work", "Reduced material variance by 8%", "Supported project delivery 2 weeks ahead"),
        avoid_terms=("react", "frontend roadmap", "backend", "api", "dsa", "github"),
    ),
    RoleProfile(
        role="Lawyer / Legal Professional",
        industry="Legal",
        career_category="Legal",
        sub_domain="Legal Practice",
        keywords=("lawyer", "advocate", "legal", "litigation", "contract", "compliance", "case law", "court", "drafting", "research"),
        skills=("legal research", "contract drafting", "litigation support", "compliance", "case analysis", "negotiation", "legal writing"),
        improvements=(
            "Add practice area, case exposure, drafting work, research depth, client interaction, and court experience.",
            "Quantify contract volume, case support, compliance reviews, or turnaround time where appropriate.",
            "Highlight bar registration, legal internships, publications, moot courts, and drafting samples if suitable.",
        ),
        certifications=("Bar Council registration", "Contract law certification", "Compliance certification", "IP law certification"),
        roadmap=(
            "Clarify practice area: corporate, litigation, IP, compliance, tax, labor, or criminal law.",
            "Add drafting, research, case, client, and court exposure with outcomes.",
            "Strengthen certifications, publications, and writing samples.",
            "Tailor resume to firm, in-house, compliance, or litigation roles.",
        ),
        metric_examples=("Reviewed 40+ contracts", "Reduced contract turnaround by 20%", "Supported 15+ litigation matters"),
        avoid_terms=TECH_AVOID_FOR_NON_TECH,
    ),
    RoleProfile(
        role="Researcher",
        industry="Research & Academia",
        career_category="Research",
        sub_domain="Academic / Applied Research",
        keywords=("research", "publication", "journal", "conference", "experiment", "thesis", "laboratory", "survey", "data collection", "analysis"),
        skills=("literature review", "research design", "data analysis", "academic writing", "experimentation", "publication", "presentation"),
        improvements=(
            "Add research question, methodology, sample size, tools, findings, publications, and conference outcomes.",
            "Quantify datasets, experiments, participants, citations, grants, or lab outputs where truthful.",
            "Highlight papers, posters, thesis work, research tools, and collaboration.",
        ),
        certifications=("Research methodology", "Good Clinical Practice", "Data analysis certificate", "Academic writing certificate"),
        roadmap=(
            "Clarify research domain, methods, tools, and output quality.",
            "Add publications, posters, datasets, experiments, and measurable findings.",
            "Strengthen methodology, analysis, writing, and presentation evidence.",
            "Tailor resume to research assistant, fellow, PhD, or industry research roles.",
        ),
        metric_examples=("Analyzed 10,000+ records", "Presented findings at 2 conferences", "Collected data from 300 participants"),
        portfolio_terms=("publication", "google scholar", "orcid", "researchgate"),
        avoid_terms=("frontend roadmap", "backend roadmap", "dsa"),
    ),
    RoleProfile(
        role="Business Professional",
        industry="Business",
        career_category="Business Operations",
        sub_domain="Operations / Strategy",
        keywords=("business", "operations", "strategy", "stakeholder", "process", "sales", "client", "project management", "requirements", "analysis"),
        skills=("stakeholder management", "process improvement", "project management", "business analysis", "communication", "reporting", "operations"),
        improvements=(
            "Add business outcomes such as revenue, cost, process time, customer satisfaction, or operational efficiency.",
            "Clarify ownership across stakeholders, requirements, reporting, projects, and process improvements.",
            "Use concise action-result bullets instead of task-only descriptions.",
        ),
        certifications=("Project Management Professional", "Certified Business Analysis Professional", "Lean Six Sigma", "Scrum Master"),
        roadmap=(
            "Clarify business function and target role.",
            "Add measurable process, revenue, customer, or stakeholder outcomes.",
            "Strengthen project management, analysis, reporting, and communication evidence.",
            "Tailor resume to operations, analyst, sales, or strategy postings.",
        ),
        metric_examples=("Reduced process time by 25%", "Supported INR 50 lakh sales pipeline", "Improved customer response time by 30%"),
        avoid_terms=("react", "backend roadmap", "docker", "kubernetes", "dsa"),
    ),
)


GENERAL_PROFILE = RoleProfile(
    role="General Professional",
    industry="General",
    career_category="Career",
    sub_domain="Transferable Skills",
    keywords=("experience", "skills", "education", "project", "achievement", "communication", "management"),
    skills=("communication", "problem solving", "teamwork", "documentation", "planning", "analysis"),
    improvements=(
        "Clarify the target role and align the resume summary, skills, and achievements to that direction.",
        "Add measurable outcomes for responsibilities, projects, academic work, or internships.",
        "Group skills into clear categories so recruiters can scan the profile quickly.",
    ),
    certifications=("Role-relevant certification", "Communication or project management certificate"),
    roadmap=(
        "Define one target role and rewrite the summary around it.",
        "Add measurable achievements and stronger section headings.",
        "Build one proof item such as project, portfolio, case work, or certification.",
        "Tailor keywords for each application before submitting.",
    ),
    metric_examples=("Improved process speed by 20%", "Led team of 5 members", "Reduced errors by 30%"),
)


SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "Summary": ("summary", "profile", "objective", "career objective", "professional summary", "about me"),
    "Skills": ("skills", "technical skills", "core skills", "key skills", "competencies", "tools"),
    "Projects": ("projects", "academic projects", "personal projects", "major projects", "case studies", "portfolio projects", "hackathons"),
    "Experience": ("experience", "work experience", "professional experience", "employment", "internship", "internships", "freelance", "practical training"),
    "Education": ("education", "academic background", "qualification", "qualifications", "academics"),
    "Achievements": ("achievements", "awards", "honors", "accomplishments"),
    "Certifications": ("certifications", "certificates", "licenses", "licences", "training"),
    "Languages": ("languages", "language"),
    "Portfolio": ("portfolio", "links", "profiles", "publications", "online presence"),
}


ACTION_VERBS = (
    "achieved",
    "analyzed",
    "built",
    "coordinated",
    "created",
    "delivered",
    "designed",
    "developed",
    "diagnosed",
    "directed",
    "documented",
    "drove",
    "evaluated",
    "executed",
    "facilitated",
    "improved",
    "increased",
    "led",
    "managed",
    "optimized",
    "performed",
    "planned",
    "reduced",
    "resolved",
    "supported",
    "taught",
    "treated",
)


CONTACT_PATTERNS = {
    "email": re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])"),
    "email_candidate": re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+(?![\w.-])"),
    "phone": re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3,5}\)?[\s.-]?)?\d{3,5}[\s.-]?\d{4}"),
    "linkedin": re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/(?:in|pub|profile)/[\w./%-]+|\blinkedin\b", re.IGNORECASE),
    "github": re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w.-]+|\bgithub\b", re.IGNORECASE),
    "portfolio": re.compile(r"behance|dribbble|portfolio|figma|researchgate|orcid|google scholar|personal website|netlify\.app|vercel\.app|github\.io", re.IGNORECASE),
}

URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>()]+", re.IGNORECASE)

FRESHER_TERMS = (
    "fresher",
    "entry level",
    "entry-level",
    "student",
    "graduate",
    "recent graduate",
    "final year",
    "campus",
    "trainee",
)

EXPERIENCE_ALTERNATIVE_TERMS = (
    "internship",
    "intern",
    "academic project",
    "major project",
    "personal project",
    "capstone",
    "freelance",
    "hackathon",
    "practical training",
    "industrial training",
    "clinical rotation",
    "case study",
    "volunteer",
    "apprenticeship",
)


def analyze_resume_doctor(
    resume_text: str,
    job_description: str = "",
    local_features: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = clean_text(resume_text)
    local_features = local_features or {}
    diagnostics = diagnostics or {}

    profile = detect_profile(text, local_features)
    sections = extract_sections(text)
    contact_details = extract_contact_details(text)
    experience_alternatives = detect_experience_alternatives(text, sections)
    candidate_level = detect_candidate_level(text, sections, experience_alternatives)
    domain_skills = extract_domain_skills(text, profile)
    missing_sections = detect_missing_sections(text, sections, profile, candidate_level, experience_alternatives)
    achievements = detect_achievements(text, sections, profile)
    ats = analyze_ats(text, sections, profile, job_description, diagnostics, missing_sections)
    section_analysis = analyze_sections(
        text,
        sections,
        profile,
        local_features,
        domain_skills,
        achievements,
        contact_details,
        candidate_level,
        experience_alternatives,
    )
    grammar = analyze_grammar(text)
    scores = calculate_scores(text, sections, profile, domain_skills, achievements, ats, grammar, job_description, candidate_level, experience_alternatives)
    heatmap = build_heatmap(section_analysis, ats, achievements)
    rewrites = generate_section_rewrites(section_analysis, profile)
    personalized_improvements = domain_safe_list(profile.improvements, profile)
    roadmap = build_roadmap(profile, scores, domain_skills)
    job_match = analyze_job_description_match(text, job_description, profile, domain_skills) if job_description.strip() else {}
    apply_readiness = calculate_apply_readiness(scores, ats, missing_sections, achievements)

    strengths = collect_strengths(section_analysis, domain_skills, achievements, profile)
    gaps = collect_gaps(section_analysis, ats, missing_sections, achievements)
    recommendations = collect_recommendations(section_analysis, personalized_improvements, ats, missing_sections, profile)

    report = {
        "summary": build_summary(profile, scores, strengths, gaps),
        "match_score": scores["overall"],
        "profile": profile_to_dict(profile),
        "candidate_level": candidate_level,
        "experience_alternatives": experience_alternatives,
        "contact_details": contact_details,
        "sections": section_analysis,
        "scores": scores,
        "score_breakdown": build_score_breakdown(scores),
        "ats": ats,
        "heatmap": heatmap,
        "rewrites": rewrites,
        "achievements": achievements,
        "missing_sections": missing_sections,
        "personalized_improvements": personalized_improvements,
        "roadmap": roadmap,
        "job_match": job_match,
        "apply_readiness": apply_readiness,
        "extracted_skills": domain_skills,
        "matched_keywords": ats.get("matched_keywords", []),
        "missing_keywords": ats.get("missing_keywords", []),
        "suggested_roles": [profile.role, *[item["role"] for item in profile_to_dict(profile).get("secondary_matches", [])]],
        "strengths": strengths,
        "gaps": gaps,
        "recommendations": recommendations,
        "diagnostics": diagnostics,
    }
    return report


def detect_profile(resume_text: str, local_features: dict[str, Any] | None = None) -> RoleProfile:
    text = normalize_text(resume_text)
    top_text = normalize_text("\n".join(resume_text.splitlines()[:25]))
    scores: list[tuple[int, RoleProfile]] = []

    for profile in ROLE_PROFILES:
        score = 0
        for keyword in profile.keywords:
            occurrences = count_keyword(keyword, text)
            score += occurrences * (5 if " " in keyword else 3)
            if keyword_in_text(keyword, top_text):
                score += 10
        for skill in profile.skills:
            if keyword_in_text(skill, text):
                score += 2
        if keyword_in_text(profile.role, top_text):
            score += 18
        scores.append((score, profile))

    scores.sort(key=lambda item: item[0], reverse=True)
    if not scores or scores[0][0] <= 3:
        return GENERAL_PROFILE

    top_score, top_profile = scores[0]
    total_top = max(top_score + scores[1][0] + scores[2][0], 1) if len(scores) > 2 else max(top_score, 1)
    confidence = min(96, max(50, round(55 + (top_score / total_top) * 45)))
    return attach_confidence(top_profile, confidence, scores[1:4])


def attach_confidence(profile: RoleProfile, confidence: int, alternatives: list[tuple[int, RoleProfile]]) -> RoleProfile:
    object.__setattr__(profile, "_confidence", confidence)
    secondary = []
    max_score = alternatives[0][0] if alternatives else 1
    for score, alt_profile in alternatives:
        if score <= 2:
            continue
        alt_confidence = max(25, min(88, round((score / max(max_score, 1)) * max(confidence - 12, 35))))
        secondary.append({"role": alt_profile.role, "industry": alt_profile.industry, "confidence": alt_confidence})
    object.__setattr__(profile, "_secondary_matches", secondary)
    return profile


def profile_to_dict(profile: RoleProfile) -> dict[str, Any]:
    return {
        "industry": profile.industry,
        "role": profile.role,
        "sub_domain": profile.sub_domain,
        "career_category": profile.career_category,
        "confidence": int(getattr(profile, "_confidence", 68 if profile is GENERAL_PROFILE else 75)),
        "secondary_matches": list(getattr(profile, "_secondary_matches", [])),
        "target_skills": list(profile.skills),
        "certifications": list(profile.certifications),
        "job_query": profile.role,
    }


def extract_sections(resume_text: str) -> dict[str, dict[str, Any]]:
    lines = [line.rstrip() for line in resume_text.splitlines()]
    headings: list[tuple[int, str]] = []
    alias_to_section = {
        alias: section
        for section, aliases in SECTION_ALIASES.items()
        for alias in aliases
    }

    for index, line in enumerate(lines):
        section = identify_heading(line, alias_to_section)
        if section:
            headings.append((index, section))

    sections: dict[str, dict[str, Any]] = {}
    for pos, (line_index, section) in enumerate(headings):
        next_index = headings[pos + 1][0] if pos + 1 < len(headings) else len(lines)
        content = "\n".join(lines[line_index + 1 : next_index]).strip()
        if content:
            sections[section] = {"heading_line": line_index + 1, "text": content, "word_count": count_words(content)}

    contact_text = "\n".join(lines[:12]).strip()
    if contact_text:
        sections.setdefault("Contact", {"heading_line": 1, "text": contact_text, "word_count": count_words(contact_text)})

    infer_implicit_sections(sections, resume_text)
    return sections


def identify_heading(line: str, alias_to_section: dict[str, str]) -> str | None:
    raw = line.strip()
    if not raw or len(raw) > 48:
        return None
    normalized = re.sub(r"[:|/\-]+$", "", raw.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in alias_to_section:
        return alias_to_section[normalized]
    if len(normalized.split()) <= 4:
        for alias, section in alias_to_section.items():
            if normalized == alias or normalized.endswith(" " + alias):
                return section
    return None


def infer_implicit_sections(sections: dict[str, dict[str, Any]], resume_text: str) -> None:
    text = normalize_text(resume_text)
    experience_hint = re.search(r"\b(?:experience|intern|freelance|trainee|apprentice|worked|employment|present|current)\b", text)
    if "Experience" not in sections and experience_hint and re.search(r"\b(?:19|20)\d{2}\s*(?:-|to|present|current)", text):
        snippets = [
            line
            for line in resume_text.splitlines()
            if re.search(r"\b(?:19|20)\d{2}\b|present|current", line, re.I)
            and not re.search(r"\b(?:bachelor|master|phd|mbbs|b\.?tech|mba|degree|university|college|school)\b", line, re.I)
        ]
        if snippets:
            sections["Experience"] = {"heading_line": 0, "text": "\n".join(snippets[:12]), "word_count": count_words("\n".join(snippets[:12]))}
    if "Education" not in sections and re.search(r"\b(?:bachelor|master|phd|mbbs|b\.?tech|mba|degree|university|college)\b", text):
        snippets = [line for line in resume_text.splitlines() if re.search(r"\b(?:bachelor|master|phd|mbbs|b\.?tech|mba|degree|university|college)\b", line, re.I)]
        sections["Education"] = {"heading_line": 0, "text": "\n".join(snippets[:8]), "word_count": count_words("\n".join(snippets[:8]))}


def extract_domain_skills(resume_text: str, profile: RoleProfile) -> list[str]:
    text = normalize_text(resume_text)
    skills: list[str] = []
    for skill in profile.skills:
        if keyword_in_text(skill, text):
            skills.append(skill)

    # Include profile keywords that are useful as ATS terms, while keeping them domain scoped.
    for keyword in profile.keywords:
        if keyword not in skills and keyword_in_text(keyword, text):
            skills.append(keyword)

    return sorted(skills, key=lambda value: (value not in profile.skills, value))


def extract_contact_details(resume_text: str) -> dict[str, Any]:
    valid_emails = [match.group(0).lower() for match in CONTACT_PATTERNS["email"].finditer(resume_text or "")]
    candidates = [match.group(0).lower().strip(".,;:") for match in CONTACT_PATTERNS["email_candidate"].finditer(resume_text or "")]
    invalid_emails = [candidate for candidate in candidates if candidate not in valid_emails]
    phone_match = CONTACT_PATTERNS["phone"].search(resume_text or "")
    urls = [url.rstrip(".,);]") for url in URL_PATTERN.findall(resume_text or "")]
    live_links = [
        url
        for url in urls
        if not re.search(r"linkedin\.com|github\.com|mailto:", url, flags=re.IGNORECASE)
        and re.search(r"portfolio|project|demo|app|netlify|vercel|github\.io|behance|dribbble", url, flags=re.IGNORECASE)
    ]

    email_status = "valid" if valid_emails else "invalid" if invalid_emails else "missing"
    return {
        "email": valid_emails[0] if valid_emails else invalid_emails[0] if invalid_emails else "",
        "email_status": email_status,
        "email_label": "Email found" if email_status == "valid" else "Email found but invalid" if email_status == "invalid" else "Email missing",
        "phone": phone_match.group(0).strip() if phone_match else "",
        "phone_status": "found" if phone_match else "missing",
        "linkedin": bool(CONTACT_PATTERNS["linkedin"].search(resume_text or "")),
        "github": bool(CONTACT_PATTERNS["github"].search(resume_text or "")),
        "portfolio": bool(CONTACT_PATTERNS["portfolio"].search(resume_text or "") or live_links),
        "live_project_links": live_links[:5],
        "urls": urls[:12],
    }


def detect_experience_alternatives(resume_text: str, sections: dict[str, dict[str, Any]]) -> list[str]:
    text = normalize_text(
        " ".join(
            [
                resume_text,
                sections.get("Projects", {}).get("text", ""),
                sections.get("Experience", {}).get("text", ""),
                sections.get("Education", {}).get("text", ""),
            ]
        )
    )
    labels: list[str] = []
    label_terms = {
        "Internships": ("internship", "intern"),
        "Academic projects": ("academic project", "major project", "capstone", "personal project"),
        "Freelance work": ("freelance", "client project"),
        "Hackathons": ("hackathon",),
        "Practical training": ("practical training", "industrial training", "apprenticeship"),
        "Clinical or field exposure": ("clinical rotation", "field work", "case study", "volunteer"),
    }
    for label, terms in label_terms.items():
        if any(keyword_in_text(term, text) for term in terms):
            labels.append(label)
    return labels


def detect_candidate_level(resume_text: str, sections: dict[str, dict[str, Any]], alternatives: list[str]) -> str:
    text = normalize_text(resume_text)
    explicit_years = [int(value) for value in re.findall(r"\b(\d{1,2})\+?\s*(?:years|yrs)\b", text)]
    if explicit_years and max(explicit_years) >= 2:
        return "experienced"
    has_fresher_language = any(keyword_in_text(term, text) for term in FRESHER_TERMS)
    has_formal_experience = sections.get("Experience", {}).get("word_count", 0) >= 45 and not has_fresher_language
    if has_fresher_language or (alternatives and not has_formal_experience):
        return "fresher"
    return "early-career" if alternatives else "experienced"


def detect_missing_sections(
    resume_text: str,
    sections: dict[str, dict[str, Any]],
    profile: RoleProfile,
    candidate_level: str = "experienced",
    experience_alternatives: list[str] | None = None,
) -> list[dict[str, str]]:
    text = normalize_text(resume_text)
    experience_alternatives = experience_alternatives or []
    missing: list[dict[str, str]] = []

    required = ["Contact", "Summary", "Skills", "Education"]
    if profile.career_category in {"Technology", "Creative & Product", "Research"} or candidate_level == "fresher":
        required.append("Projects")
    if profile.role not in {"General Professional"} and candidate_level != "fresher":
        required.append("Experience")
    required.append("Achievements")

    for section in required:
        if section not in sections or sections.get(section, {}).get("word_count", 0) < 8:
            missing.append(
                {
                    "section": section,
                    "severity": "high" if section in {"Contact", "Skills", "Experience", "Education"} else "medium",
                    "suggestion": missing_section_suggestion(section, profile),
                }
            )

    if candidate_level == "fresher" and not experience_alternatives and sections.get("Projects", {}).get("word_count", 0) < 20:
        missing.append(
            {
                "section": "Experience alternatives",
                "severity": "medium",
                "suggestion": "Add internships, academic projects, freelance work, hackathons, practical training, or major projects as proof of readiness.",
            }
        )

    if profile.industry in {"Healthcare", "Education", "Finance", "Legal", "Engineering"} and "Certifications" not in sections:
        missing.append(
            {
                "section": "Certifications",
                "severity": "medium",
                "suggestion": "Add role-relevant licenses, certifications, training, or professional registration.",
            }
        )

    if should_have_portfolio(profile) and not any(keyword_in_text(term, text) for term in profile.portfolio_terms):
        missing.append(
            {
                "section": "Portfolio",
                "severity": "medium",
                "suggestion": portfolio_suggestion(profile),
            }
        )

    if profile.industry == "Software & IT" and not CONTACT_PATTERNS["github"].search(resume_text):
        missing.append(
            {
                "section": "GitHub / Code Link",
                "severity": "low",
                "suggestion": "Add a GitHub or project link only if the repositories are clean, relevant, and recruiter-ready.",
            }
        )

    contact_details = extract_contact_details(resume_text)
    if profile.industry == "Software & IT" and not contact_details.get("live_project_links"):
        missing.append(
            {
                "section": "Live Project Link",
                "severity": "low",
                "suggestion": "Add a live project/demo link only when it is polished, working, and relevant to the target role.",
            }
        )

    if not CONTACT_PATTERNS["linkedin"].search(resume_text):
        missing.append(
            {
                "section": "LinkedIn",
                "severity": "low",
                "suggestion": "Add a current LinkedIn profile so recruiters can verify your professional background.",
            }
        )

    return missing


def missing_section_suggestion(section: str, profile: RoleProfile) -> str:
    if section == "Summary":
        return f"Add a 2-3 line summary focused on {profile.role} strengths and target context."
    if section == "Skills":
        return f"Group {profile.role} skills into clear categories instead of a scattered list."
    if section == "Projects":
        if profile.industry == "Healthcare":
            return "Add clinical cases, rotations, audits, research, or patient-care initiatives instead of software projects."
        if profile.industry == "Education":
            return "Add teaching initiatives, lesson plans, curriculum work, or student improvement projects."
        return "Add role-relevant projects or case work with problem, action, tools, and measurable result."
    if section == "Achievements":
        return "Add measurable achievements with numbers, scale, outcomes, awards, or recognitions."
    if section == "Experience":
        return "Add responsibilities and achievements for internships, work, clinical, teaching, field, or practical experience."
    if section == "Education":
        return "Add degree, institution, dates, and relevant coursework or academic achievements."
    return f"Add a clear {section} section."


def should_have_portfolio(profile: RoleProfile) -> bool:
    return profile.role in {"UI/UX Designer", "Marketing Professional", "Researcher"} or bool(profile.portfolio_terms)


def portfolio_suggestion(profile: RoleProfile) -> str:
    if profile.role == "UI/UX Designer":
        return "Add Behance, Figma, Dribbble, or portfolio case-study links."
    if profile.role == "Marketing Professional":
        return "Add campaign samples, portfolio, or website links when available."
    if profile.role == "Researcher":
        return "Add publications, ORCID, Google Scholar, or ResearchGate links where relevant."
    return "Add a role-relevant portfolio or proof-of-work link."


def detect_achievements(resume_text: str, sections: dict[str, dict[str, Any]], profile: RoleProfile) -> dict[str, Any]:
    lines = [line.strip(" -\u2022\t") for line in resume_text.splitlines() if line.strip()]
    bullet_like = [
        line
        for line in lines
        if len(line.split()) >= 5
        and (
            re.match(r"^[\-*\u2022]", line)
            or any(keyword_in_text(verb, line) for verb in ACTION_VERBS)
        )
    ]
    metrics = extract_metrics(resume_text)
    weak_bullets = [line for line in bullet_like if not extract_metrics(line)][:8]

    suggestions = []
    for example in profile.metric_examples[:3]:
        suggestions.append(f"If truthful, add a metric like: {example}.")
    if weak_bullets:
        suggestions.append("Rewrite task-only bullets into action + scope + result; add numbers only when you can prove them.")

    return {
        "metric_count": len(metrics),
        "metrics": metrics[:12],
        "weak_bullets": weak_bullets,
        "weak_bullet_count": len(weak_bullets),
        "suggestions": domain_safe_list(suggestions, profile),
    }


def analyze_ats(
    resume_text: str,
    sections: dict[str, dict[str, Any]],
    profile: RoleProfile,
    job_description: str,
    diagnostics: dict[str, Any],
    missing_sections: list[dict[str, str]],
) -> dict[str, Any]:
    target_keywords = build_target_keywords(profile, job_description)
    text = normalize_text(resume_text)
    matched = [keyword for keyword in target_keywords if keyword_in_text(keyword, text)]
    missing = [keyword for keyword in target_keywords if keyword not in matched]
    keyword_coverage = round((len(matched) / len(target_keywords)) * 100) if target_keywords else 0

    issues: list[dict[str, str]] = []
    passed: list[str] = []

    if diagnostics.get("table_count", 0):
        issues.append(
            {
                "issue": "Tables found",
                "reason": "Many ATS parsers read tables out of order or skip cells.",
                "suggestion": "Use plain section headings and simple text columns for critical content.",
                "severity": "high",
            }
        )
    else:
        passed.append("No PDF tables detected.")

    if diagnostics.get("image_count", 0):
        issues.append(
            {
                "issue": "Images or image-heavy content found",
                "reason": "ATS systems can ignore text embedded in images.",
                "suggestion": "Keep core resume content as selectable text, with images used only when nonessential.",
                "severity": "high",
            }
        )
    else:
        passed.append("No critical image dependency detected.")

    if diagnostics.get("multi_column_pages", 0):
        issues.append(
            {
                "issue": "Possible multi-column layout",
                "reason": "Multi-column resumes can be parsed in the wrong reading order.",
                "suggestion": "Prefer one-column layout for ATS submissions.",
                "severity": "medium",
            }
        )

    if has_symbol_noise(resume_text):
        issues.append(
            {
                "issue": "Decorative symbols or icons",
                "reason": "Icons and nonstandard bullets can reduce parser accuracy.",
                "suggestion": "Use standard text labels for phone, email, links, and section headings.",
                "severity": "medium",
            }
        )

    for item in missing_sections[:5]:
        issues.append(
            {
                "issue": f"Missing or weak {item['section']} section",
                "reason": "ATS and recruiters rely on clear section labels to classify content.",
                "suggestion": item["suggestion"],
                "severity": item["severity"],
            }
        )

    if keyword_coverage < 55:
        issues.append(
            {
                "issue": "Low role keyword coverage",
                "reason": f"The resume matches {keyword_coverage}% of detected {profile.role} keywords.",
                "suggestion": "Add only truthful role-specific terms from your actual skills, training, and experience.",
                "severity": "high",
            }
        )
    else:
        passed.append("Role keyword coverage is readable.")

    score = 100
    severity_penalty = {"high": 16, "medium": 9, "low": 4}
    for issue in issues:
        score -= severity_penalty.get(issue.get("severity", "medium"), 8)
    score = clamp(round((score * 0.55) + (keyword_coverage * 0.45)), 0, 100)

    return {
        "score": score,
        "percentage": keyword_coverage,
        "keyword_coverage": keyword_coverage,
        "matched_keywords": matched[:30],
        "missing_keywords": missing[:30],
        "target_keywords": target_keywords,
        "issues": domain_safe_issues(issues, profile),
        "passed": passed,
    }


def build_target_keywords(profile: RoleProfile, job_description: str = "") -> list[str]:
    keywords = list(profile.skills) + list(profile.keywords[:12])
    if job_description.strip():
        keywords.extend(extract_meaningful_keywords(job_description, limit=18))
    deduped: list[str] = []
    for keyword in keywords:
        value = normalize_text(keyword)
        if value and value not in deduped and not is_avoid_term(value, profile):
            deduped.append(value)
    return deduped[:40]


def analyze_sections(
    resume_text: str,
    sections: dict[str, dict[str, Any]],
    profile: RoleProfile,
    local_features: dict[str, Any],
    domain_skills: list[str],
    achievements: dict[str, Any],
    contact_details: dict[str, Any] | None = None,
    candidate_level: str = "experienced",
    experience_alternatives: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    section_names = ["Contact", "Summary", "Skills", "Experience", "Projects", "Education", "Achievements", "Certifications", "Languages", "Portfolio"]
    analysis: dict[str, dict[str, Any]] = {}
    contact_details = contact_details or extract_contact_details(resume_text)
    experience_alternatives = experience_alternatives or []

    for name in section_names:
        section = sections.get(name, {})
        text = section.get("text", "")
        strengths: list[str] = []
        weaknesses: list[str] = []
        suggestions: list[str] = []
        score = 45 if text else 20

        if name == "Contact":
            score, strengths, weaknesses, suggestions = analyze_contact(resume_text, profile, contact_details)
        elif name == "Summary":
            score, strengths, weaknesses, suggestions = analyze_summary(text, profile)
        elif name == "Skills":
            score, strengths, weaknesses, suggestions = analyze_skills_section(text, profile, domain_skills)
        elif name == "Experience":
            score, strengths, weaknesses, suggestions = analyze_experience_section(text, resume_text, profile, candidate_level, experience_alternatives)
        elif name == "Projects":
            score, strengths, weaknesses, suggestions = analyze_projects_section(text, profile)
        elif name == "Education":
            score, strengths, weaknesses, suggestions = analyze_education_section(text, resume_text, profile)
        elif name == "Achievements":
            score, strengths, weaknesses, suggestions = analyze_achievements_section(text, achievements, profile)
        elif name == "Certifications":
            score, strengths, weaknesses, suggestions = analyze_certifications_section(text, profile)
        elif name == "Languages":
            score, strengths, weaknesses, suggestions = analyze_languages_section(text, profile)
        elif name == "Portfolio":
            score, strengths, weaknesses, suggestions = analyze_portfolio_section(text, resume_text, profile)

        score = clamp(score, 0, 100)
        analysis[name] = {
            "present": bool(text) if name != "Contact" else has_contact_info(resume_text, contact_details),
            "score": score,
            "heat": heat_level(score),
            "strengths": domain_safe_list(strengths, profile),
            "weaknesses": domain_safe_list(weaknesses, profile),
            "suggestions": domain_safe_list(suggestions, profile),
            "excerpt": compact_excerpt(text or sections.get("Contact", {}).get("text", ""), 420),
        }

    return analysis


def analyze_contact(resume_text: str, profile: RoleProfile, contact_details: dict[str, Any] | None = None) -> tuple[int, list[str], list[str], list[str]]:
    contact_details = contact_details or extract_contact_details(resume_text)
    strengths: list[str] = []
    weaknesses: list[str] = []
    suggestions: list[str] = []
    score = 15

    if contact_details.get("email_status") == "valid":
        strengths.append("Includes a valid-looking email.")
        score += 18
    elif contact_details.get("email_status") == "invalid":
        weaknesses.append("Email found but invalid.")
        suggestions.append("Fix the email format so recruiters and ATS parsers can read it.")
        score += 6
    else:
        weaknesses.append("Email is missing.")
        suggestions.append("Add a current professional email.")

    if contact_details.get("phone_status") == "found":
        strengths.append("Includes phone number.")
        score += 18
    else:
        weaknesses.append("Phone number is missing.")
        suggestions.append("Add a reachable phone number with country code if applying internationally.")

    if contact_details.get("linkedin"):
        strengths.append("Includes LinkedIn.")
        score += 12
    else:
        weaknesses.append("LinkedIn is missing.")
        suggestions.append("Add a current LinkedIn profile so recruiters can verify your background.")

    if profile.industry == "Software & IT":
        if contact_details.get("github"):
            strengths.append("Includes GitHub/code profile link.")
            score += 12
        else:
            weaknesses.append("GitHub or code link is missing.")
            suggestions.append("Add a GitHub or project link if it represents your best work.")
    elif should_have_portfolio(profile):
        if contact_details.get("portfolio"):
            strengths.append("Includes portfolio/proof link.")
            score += 12
        else:
            weaknesses.append("Portfolio/proof link is missing.")
            suggestions.append(portfolio_suggestion(profile))
    else:
        score += 8

    if profile.industry == "Software & IT":
        if contact_details.get("live_project_links"):
            strengths.append("Includes live project/demo link.")
            score += 5
        else:
            suggestions.append("Add a live project or portfolio link when it is polished and truthful.")

    return score, strengths, weaknesses, suggestions


def analyze_summary(text: str, profile: RoleProfile) -> tuple[int, list[str], list[str], list[str]]:
    if not text:
        return 20, [], ["Summary is missing."], [f"Add a concise summary focused on {profile.role} experience, strengths, and target direction."]
    words = count_words(text)
    strengths = ["Summary section is present."]
    weaknesses: list[str] = []
    suggestions: list[str] = []
    score = 60
    if 25 <= words <= 90:
        strengths.append("Summary length is recruiter-friendly.")
        score += 20
    else:
        weaknesses.append("Summary length is either too thin or too long.")
        suggestions.append("Keep the summary to 2-3 specific lines.")
    if any(keyword_in_text(keyword, text) for keyword in profile.keywords[:8]):
        strengths.append("Summary contains role-relevant language.")
        score += 15
    else:
        weaknesses.append("Summary does not clearly signal the detected role.")
        suggestions.append(f"Mention {profile.role} focus and strongest domain evidence.")
    return score, strengths, weaknesses, suggestions


def analyze_skills_section(text: str, profile: RoleProfile, domain_skills: list[str]) -> tuple[int, list[str], list[str], list[str]]:
    strengths: list[str] = []
    weaknesses: list[str] = []
    suggestions: list[str] = []
    coverage = round((len(domain_skills) / max(len(profile.skills), 1)) * 100)
    score = min(95, 30 + coverage)
    if text:
        strengths.append("Skills section is present.")
    else:
        weaknesses.append("Skills section is missing or not clearly labeled.")
        suggestions.append("Create a clear skills section with role-specific categories.")
        score -= 20
    if domain_skills:
        strengths.append("Detected role-relevant skills: " + ", ".join(domain_skills[:8]) + ".")
    if coverage < 45:
        weaknesses.append(f"Limited visible {profile.role} skill coverage.")
        suggestions.append("Add truthful domain skills from your training, tools, and hands-on work.")
    return score, strengths, weaknesses, suggestions


def analyze_experience_section(
    text: str,
    resume_text: str,
    profile: RoleProfile,
    candidate_level: str = "experienced",
    experience_alternatives: list[str] | None = None,
) -> tuple[int, list[str], list[str], list[str]]:
    target_text = text or resume_text
    experience_alternatives = experience_alternatives or []
    strengths: list[str] = []
    weaknesses: list[str] = []
    suggestions: list[str] = []
    score = 35 if text else 20
    if text:
        strengths.append("Experience section is visible.")
    elif candidate_level == "fresher" and experience_alternatives:
        strengths.append("Fresher-friendly experience evidence is visible through " + ", ".join(experience_alternatives[:3]) + ".")
        suggestions.append("Label this evidence clearly as Internships, Projects, Training, Hackathons, or Freelance Work.")
        score = 62
    else:
        weaknesses.append("Experience section is missing or hard to parse.")
    if re.search(r"\b(?:19|20)\d{2}\b|present|current", target_text, re.I):
        strengths.append("Includes dates or timeline evidence.")
        score += 15
    elif candidate_level == "fresher" and experience_alternatives:
        suggestions.append("Add dates for internships, academic projects, practical training, or hackathons.")
        score += 6
    else:
        weaknesses.append("Dates or timeline evidence are weak.")
        suggestions.append("Add month/year or year ranges for experience, internships, rotations, or practice.")
    verbs = [verb for verb in ACTION_VERBS if keyword_in_text(verb, target_text)]
    if len(verbs) >= 3:
        strengths.append("Uses action-oriented language.")
        score += 15
    else:
        weaknesses.append("Experience bullets need stronger action verbs.")
        suggestions.append("Start bullets with action verbs and end with outcomes.")
    if extract_metrics(target_text):
        strengths.append("Includes measurable impact.")
        score += 25
    else:
        weaknesses.append("Experience lacks measurable outcomes.")
        suggestions.append("Add truthful numbers for scale, quality, speed, or impact if you have evidence.")
    return score, strengths, weaknesses, suggestions


def analyze_projects_section(text: str, profile: RoleProfile) -> tuple[int, list[str], list[str], list[str]]:
    if not text:
        label = "projects"
        if profile.industry == "Healthcare":
            label = "clinical cases, audits, research, or rotations"
        elif profile.industry == "Education":
            label = "teaching initiatives or curriculum work"
        return 25, [], [f"Role-relevant {label} are missing or not labeled."], [missing_section_suggestion("Projects", profile)]

    strengths = ["Role evidence section is present."]
    weaknesses: list[str] = []
    suggestions: list[str] = []
    score = 55
    if any(keyword_in_text(skill, text) for skill in profile.skills):
        strengths.append("Includes relevant tools or methods.")
        score += 15
    else:
        weaknesses.append("The section needs clearer role-relevant tools, methods, or context.")
        suggestions.append("Mention tools, methods, constraints, and your specific contribution.")
    if extract_metrics(text):
        strengths.append("Includes measurable project/case impact.")
        score += 20
    else:
        weaknesses.append("Project or case work lacks measurable outcomes.")
        suggestions.append("If truthful, add project scope, users, accuracy, speed, grade, adoption, or other measurable proof.")
    if len(text.splitlines()) >= 3:
        score += 8
    return score, strengths, weaknesses, suggestions


def analyze_education_section(text: str, resume_text: str, profile: RoleProfile) -> tuple[int, list[str], list[str], list[str]]:
    source = text or resume_text
    strengths: list[str] = []
    weaknesses: list[str] = []
    suggestions: list[str] = []
    score = 40 if text else 25
    if re.search(r"\b(?:bachelor|master|phd|doctorate|mbbs|md|b\.?tech|m\.?tech|b\.?ed|mba|degree|diploma)\b", source, re.I):
        strengths.append("Degree or qualification is visible.")
        score += 25
    else:
        weaknesses.append("Degree or qualification is not clear.")
        suggestions.append("Add degree/qualification, institution, and completion year.")
    if re.search(r"\b(?:university|college|institute|school)\b", source, re.I):
        strengths.append("Institution is visible.")
        score += 15
    if re.search(r"\b(?:19|20)\d{2}\b|cgpa|gpa|percentage|grade", source, re.I):
        strengths.append("Includes year or academic performance detail.")
        score += 10
    else:
        suggestions.append("Add completion year, relevant coursework, or grade details when useful.")
    return score, strengths, weaknesses, suggestions


def analyze_achievements_section(text: str, achievements: dict[str, Any], profile: RoleProfile) -> tuple[int, list[str], list[str], list[str]]:
    score = 30 if not text else 55
    strengths: list[str] = []
    weaknesses: list[str] = []
    suggestions: list[str] = []
    if text:
        strengths.append("Achievements section is present.")
    else:
        weaknesses.append("Achievements are not separated clearly.")
    if achievements.get("metric_count", 0) >= 3:
        strengths.append("Multiple measurable outcomes detected.")
        score += 30
    elif achievements.get("metric_count", 0):
        strengths.append("Some measurable outcomes detected.")
        score += 15
    else:
        weaknesses.append("No measurable achievements detected.")
        suggestions.extend(achievements.get("suggestions", [])[:2])
    return score, strengths, weaknesses, suggestions


def analyze_certifications_section(text: str, profile: RoleProfile) -> tuple[int, list[str], list[str], list[str]]:
    if text:
        return 80, ["Certification or training section is present."], [], ["Keep certifications current and relevant to the target role."]
    if profile.industry in {"Healthcare", "Education", "Finance", "Legal", "Engineering"}:
        return 35, [], ["Certifications/licenses are missing or not labeled."], ["Add relevant items such as " + ", ".join(profile.certifications[:3]) + "."]
    return 65, [], [], ["Add certifications only when they strengthen the target role."]


def analyze_languages_section(text: str, profile: RoleProfile) -> tuple[int, list[str], list[str], list[str]]:
    if text:
        return 85, ["Languages section is present."], [], []
    if profile.industry in {"Healthcare", "Education", "Legal", "Business"}:
        return 45, [], ["Languages are not listed."], ["Add languages if they are relevant for patients, students, clients, or regional roles."]
    return 70, [], [], ["Add languages only if relevant to the job or location."]


def analyze_portfolio_section(text: str, resume_text: str, profile: RoleProfile) -> tuple[int, list[str], list[str], list[str]]:
    has_portfolio = bool(CONTACT_PATTERNS["portfolio"].search(resume_text))
    if should_have_portfolio(profile):
        if text or has_portfolio:
            return 88, ["Portfolio or proof-of-work link is visible."], [], ["Ensure links show your strongest role-relevant work."]
        return 30, [], ["Portfolio/proof link is missing."], [portfolio_suggestion(profile)]
    if profile.industry == "Software & IT" and CONTACT_PATTERNS["github"].search(resume_text):
        return 82, ["Code/project link is visible."], [], []
    return 70, [], [], []


def analyze_grammar(resume_text: str) -> dict[str, Any]:
    words = count_words(resume_text)
    sentences = re.split(r"[.!?]\s+", resume_text)
    long_sentences = [sentence for sentence in sentences if count_words(sentence) > 38]
    repeated_spaces = len(re.findall(r" {2,}", resume_text))
    first_person = len(re.findall(r"\b(?:i|me|my|mine)\b", resume_text, flags=re.I))
    typo_like = len(re.findall(r"\b\w*(?:teh|recieve|adress|responsiblity|manger|experiance)\w*\b", resume_text, flags=re.I))
    score = 100 - min(35, len(long_sentences) * 4) - min(15, repeated_spaces * 2) - min(15, first_person * 3) - min(25, typo_like * 8)
    flags = []
    if long_sentences:
        flags.append("Some sentences are too long for quick recruiter scanning.")
    if first_person:
        flags.append("Avoid first-person pronouns in resume bullets.")
    if typo_like:
        flags.append("Possible spelling issues detected.")
    return {"score": clamp(score, 0, 100), "flags": flags, "word_count": words}


def calculate_scores(
    resume_text: str,
    sections: dict[str, dict[str, Any]],
    profile: RoleProfile,
    domain_skills: list[str],
    achievements: dict[str, Any],
    ats: dict[str, Any],
    grammar: dict[str, Any],
    job_description: str,
    candidate_level: str = "experienced",
    experience_alternatives: list[str] | None = None,
) -> dict[str, int]:
    contact_details = extract_contact_details(resume_text)
    if has_contact_info(resume_text, contact_details):
        contact_score = 100
    elif contact_details.get("email_status") == "invalid" or contact_details.get("phone_status") == "found":
        contact_score = 58
    else:
        contact_score = 35
    skills_score = clamp(round((len(domain_skills) / max(len(profile.skills), 1)) * 100), 0, 100)
    formatting_score = infer_formatting_score(ats)
    experience_alternatives = experience_alternatives or []
    project_score = 80 if sections.get("Projects", {}).get("word_count", 0) >= 25 else 45
    if extract_metrics(sections.get("Projects", {}).get("text", "")):
        project_score += 12
    experience_score = 80 if sections.get("Experience", {}).get("word_count", 0) >= 30 else 45
    if candidate_level == "fresher" and experience_score < 70:
        experience_score = 70 if experience_alternatives or project_score >= 65 else 55
    if extract_metrics(sections.get("Experience", {}).get("text", resume_text)):
        experience_score += 12
    impact_score = clamp(25 + min(achievements.get("metric_count", 0) * 12, 60) - min(achievements.get("weak_bullet_count", 0) * 3, 20), 0, 100)
    keyword_score = ats.get("keyword_coverage", 0)
    ats_score = ats.get("score", 0)
    grammar_score = grammar.get("score", 0)

    overall = round(
        skills_score * 0.16
        + ats_score * 0.16
        + formatting_score * 0.11
        + grammar_score * 0.09
        + project_score * 0.12
        + experience_score * 0.14
        + impact_score * 0.13
        + keyword_score * 0.09
    )

    # A resume without contact or clear text should not look ready.
    if contact_score < 60:
        overall = min(overall, 62)
    if count_words(resume_text) < 120:
        overall = min(overall, 58)

    return {
        "overall": clamp(overall, 0, 100),
        "contact": contact_score,
        "skills": skills_score,
        "ats": ats_score,
        "formatting": formatting_score,
        "grammar": grammar_score,
        "projects": clamp(project_score, 0, 100),
        "experience": clamp(experience_score, 0, 100),
        "impact": impact_score,
        "keywords": keyword_score,
    }


def infer_formatting_score(ats: dict[str, Any]) -> int:
    score = 100
    for issue in ats.get("issues", []):
        if issue.get("issue") in {"Tables found", "Images or image-heavy content found"}:
            score -= 18
        elif issue.get("issue") == "Possible multi-column layout":
            score -= 12
        elif "symbols" in issue.get("issue", "").lower():
            score -= 8
    return clamp(score, 0, 100)


def build_score_breakdown(scores: dict[str, int]) -> dict[str, dict[str, Any]]:
    labels = {
        "skills": "Skills Score",
        "ats": "ATS Score",
        "formatting": "Formatting Score",
        "grammar": "Grammar Score",
        "projects": "Project Score",
        "experience": "Experience Score",
        "impact": "Impact Score",
        "keywords": "Keyword Score",
    }
    return {
        key: {"label": label, "points": value, "max_points": 100, "percentage": value}
        for key, label in labels.items()
        for value in [scores.get(key, 0)]
    }


def build_heatmap(section_analysis: dict[str, dict[str, Any]], ats: dict[str, Any], achievements: dict[str, Any]) -> list[dict[str, str]]:
    heatmap: list[dict[str, str]] = []
    for section, data in section_analysis.items():
        if not data.get("present") and section not in {"Languages", "Portfolio"}:
            level = "red"
            message = f"{section} is missing or difficult to parse."
        elif data.get("score", 0) >= 75:
            level = "green"
            message = f"{section} looks strong."
        elif data.get("score", 0) >= 50:
            level = "yellow"
            message = f"{section} needs sharper evidence."
        else:
            level = "red"
            message = f"{section} is weak."
        heatmap.append({"section": section, "level": level, "message": message, "excerpt": data.get("excerpt", "")})

    for issue in ats.get("issues", [])[:4]:
        heatmap.append({"section": "ATS", "level": "red" if issue.get("severity") == "high" else "yellow", "message": issue["issue"], "excerpt": issue["suggestion"]})
    if achievements.get("weak_bullets"):
        heatmap.append({"section": "Achievements", "level": "yellow", "message": "Some bullets lack measurable outcomes.", "excerpt": achievements["weak_bullets"][0]})
    return heatmap


def generate_section_rewrites(section_analysis: dict[str, dict[str, Any]], profile: RoleProfile) -> list[dict[str, Any]]:
    rewrites: list[dict[str, Any]] = []
    for section_name in ("Summary", "Experience", "Projects", "Achievements", "Skills"):
        data = section_analysis.get(section_name, {})
        before = first_rewrite_candidate(data.get("excerpt", ""))
        if not before and section_name == "Summary":
            before = f"{profile.role} candidate with verified role-relevant skills and proof of work."
        if not before:
            continue
        rewrites.append(
            {
                "section": section_name,
                "before": before,
                "versions": rewrite_text(before, section_name, profile),
            }
        )
    return rewrites


def rewrite_text(before: str, section: str, profile: RoleProfile) -> list[str]:
    cleaned = re.sub(r"\s+", " ", before).strip(" -")
    skills = ", ".join(profile.skills[:3]) or "role-relevant skills"
    has_metric = bool(extract_metrics(cleaned))
    if section == "Summary":
        return [
            f"{profile.role} candidate focused on {profile.sub_domain.lower()}, with hands-on evidence in {skills} and a clear target toward {profile.role} roles.",
            f"{profile.career_category} profile aligned to {profile.role} opportunities, highlighting {skills} and resume evidence that is easy for recruiters to verify.",
        ]
    if section == "Skills":
        return [
            f"Group skills by category: Core {profile.role} skills, tools/platforms, domain methods, and supporting skills.",
            f"Lead with the strongest verified skills: {skills}. Keep only tools you can discuss confidently in an interview.",
        ]
    if section == "Projects":
        result_part = "with the measurable result already shown in the resume" if has_metric else "with a truthful result such as users, scope, accuracy, speed, grade, or deployment status if available"
        return [
            f"Built {profile.sub_domain.lower()} work using {skills}; clarify the problem, your contribution, technology used, and {result_part}.",
            f"Rewrite as: action + technology + scope + outcome. Keep the outcome factual and avoid adding numbers unless the project already proves them.",
        ]
    if section == "Experience":
        return [
            f"Describe the work as action + responsibility + tools/methods + result. For fresher profiles, internships, training, and major projects can be framed as experience alternatives.",
            f"{cleaned.rstrip('.')}; tighten this by naming your ownership, context, and a truthful result or learning outcome.",
        ]
    return [
        f"Rewrite with action, scope, and result while staying within verified {profile.sub_domain.lower()} evidence.",
        f"{cleaned.rstrip('.')}; clarified scope, action, tools, and result to make the contribution easier for recruiters to evaluate.",
        f"Add a metric only if truthful; otherwise state the observable result, deliverable, quality bar, or learning outcome.",
    ]


def first_rewrite_candidate(excerpt: str) -> str:
    for line in excerpt.splitlines():
        candidate = line.strip(" -\u2022\t")
        if count_words(candidate) >= 4:
            return candidate[:220]
    return excerpt[:220].strip()


def build_roadmap(profile: RoleProfile, scores: dict[str, int], domain_skills: list[str]) -> dict[str, Any]:
    missing_skills = [skill for skill in profile.skills if skill not in domain_skills][:6]
    weeks = []
    for index, focus in enumerate(profile.roadmap[:4], start=1):
        tasks = [focus]
        if index == 2 and missing_skills:
            tasks.append("Close visible skill gaps: " + ", ".join(missing_skills[:4]) + ".")
        if index == 3:
            tasks.append("Add one measurable proof point only where you can verify the number.")
        if index == 4:
            tasks.append("Run ATS cleanup and tailor keywords for 3 target postings.")
        weeks.append({"week": f"Week {index}", "focus": focus, "tasks": domain_safe_list(tasks, profile)})
    return {
        "current_profile": f"{profile.role} candidate with {scores.get('overall', 0)}/100 resume readiness.",
        "target_profile": f"Stronger {profile.role} profile with clearer domain proof, ATS readability, and measurable outcomes.",
        "weeks": weeks,
    }


def analyze_job_description_match(
    resume_text: str,
    job_description: str,
    profile: RoleProfile,
    domain_skills: list[str],
) -> dict[str, Any]:
    job_keywords = build_target_keywords(profile, job_description)
    resume_text_norm = normalize_text(resume_text)
    matched = [keyword for keyword in job_keywords if keyword_in_text(keyword, resume_text_norm)]
    missing = [keyword for keyword in job_keywords if keyword not in matched]
    score = round((len(matched) / max(len(job_keywords), 1)) * 100)
    reasons = []
    if score < 70:
        reasons.append("The resume does not yet show enough keywords or evidence from the job description.")
    if missing:
        reasons.append("Important missing terms include: " + ", ".join(missing[:6]) + ".")
    if not extract_metrics(resume_text):
        reasons.append("The resume needs more measurable outcomes to increase apply readiness.")
    suggestions = []
    if missing:
        suggestions.append("Add truthful evidence for the highest-priority missing terms: " + ", ".join(missing[:5]) + ".")
    suggestions.extend(profile.improvements[:2])
    if profile.certifications:
        suggestions.append("Consider relevant credentials or training such as " + ", ".join(profile.certifications[:3]) + ".")
    return {
        "score": clamp(score, 0, 100),
        "matched": matched,
        "missing": missing,
        "reason": " ".join(reasons) if reasons else "The resume aligns well with this job description.",
        "explanation": " ".join(reasons) if reasons else "The resume aligns well with this job description. Keep tailoring wording truthfully.",
        "suggestions": domain_safe_list(suggestions, profile)[:6],
    }


def calculate_apply_readiness(
    scores: dict[str, int],
    ats: dict[str, Any],
    missing_sections: list[dict[str, str]],
    achievements: dict[str, Any],
) -> dict[str, Any]:
    readiness = round(scores["overall"] * 0.45 + scores["ats"] * 0.25 + scores["impact"] * 0.15 + scores["keywords"] * 0.15)
    ready: list[str] = []
    missing: list[str] = []
    if scores["skills"] >= 60:
        ready.append("Role-relevant skills are visible.")
    else:
        missing.append("Role skill coverage is weak.")
    if scores["ats"] >= 70:
        ready.append("ATS readability is acceptable.")
    else:
        missing.append("ATS issues need cleanup.")
    if achievements.get("metric_count", 0):
        ready.append("Some measurable impact is present.")
    else:
        missing.append("Measurable achievements are missing.")
    for item in missing_sections[:3]:
        missing.append(f"Missing or weak {item['section']} section.")
    return {"score": clamp(readiness, 0, 100), "ready": ready, "missing": list(dict.fromkeys(missing))}


def build_summary(profile: RoleProfile, scores: dict[str, int], strengths: list[str], gaps: list[str]) -> str:
    return (
        f"Detected a {profile.role} profile in {profile.industry} with {profile_to_dict(profile)['confidence']}% confidence. "
        f"The resume scores {scores['overall']}/100 overall, with strongest signals around "
        f"{strengths[0].rstrip('.') if strengths else profile.sub_domain.lower()} and priority fixes around "
        f"{gaps[0].rstrip('.') if gaps else 'clearer measurable impact'}."
    )


def collect_strengths(
    section_analysis: dict[str, dict[str, Any]],
    domain_skills: list[str],
    achievements: dict[str, Any],
    profile: RoleProfile,
) -> list[str]:
    strengths: list[str] = []
    if domain_skills:
        strengths.append(f"Shows {profile.role} skill signals: {', '.join(domain_skills[:6])}.")
    if achievements.get("metric_count", 0):
        strengths.append(f"Includes {achievements['metric_count']} measurable outcome signal(s).")
    for section in ("Contact", "Education", "Experience", "Projects"):
        for item in section_analysis.get(section, {}).get("strengths", [])[:1]:
            strengths.append(item)
    return domain_safe_list(list(dict.fromkeys(strengths))[:8], profile)


def collect_gaps(
    section_analysis: dict[str, dict[str, Any]],
    ats: dict[str, Any],
    missing_sections: list[dict[str, str]],
    achievements: dict[str, Any],
) -> list[str]:
    gaps: list[str] = []
    for item in missing_sections[:4]:
        gaps.append(f"{item['section']} is missing or weak.")
    for issue in ats.get("issues", [])[:3]:
        gaps.append(issue["issue"] + ".")
    if achievements.get("weak_bullet_count", 0):
        gaps.append("Several bullets lack measurable outcomes.")
    for section in section_analysis.values():
        gaps.extend(section.get("weaknesses", [])[:1])
    return list(dict.fromkeys(gaps))[:10]


def collect_recommendations(
    section_analysis: dict[str, dict[str, Any]],
    personalized_improvements: list[str],
    ats: dict[str, Any],
    missing_sections: list[dict[str, str]],
    profile: RoleProfile,
) -> list[str]:
    recommendations: list[str] = []
    recommendations.extend(personalized_improvements[:4])
    recommendations.extend(item["suggestion"] for item in missing_sections[:4])
    recommendations.extend(issue["suggestion"] for issue in ats.get("issues", [])[:4])
    for section in section_analysis.values():
        recommendations.extend(section.get("suggestions", [])[:1])
    return domain_safe_list(list(dict.fromkeys(recommendations))[:12], profile)


def compare_resume_versions(
    resume_v1_text: str,
    resume_v2_text: str,
    job_description: str = "",
    diagnostics_v1: dict[str, Any] | None = None,
    diagnostics_v2: dict[str, Any] | None = None,
) -> dict[str, Any]:
    v1 = analyze_resume_doctor(resume_v1_text, job_description=job_description, diagnostics=diagnostics_v1 or {})
    v2 = analyze_resume_doctor(resume_v2_text, job_description=job_description, diagnostics=diagnostics_v2 or {})
    skills_v1 = set(v1.get("extracted_skills", []))
    skills_v2 = set(v2.get("extracted_skills", []))
    score_v1 = v1["scores"]["overall"]
    score_v2 = v2["scores"]["overall"]
    improvement = score_v2 - score_v1
    improvement_pct = round((improvement / max(score_v1, 1)) * 100)
    return {
        "v1": {"score": score_v1, "profile": v1["profile"], "skills": sorted(skills_v1)},
        "v2": {"score": score_v2, "profile": v2["profile"], "skills": sorted(skills_v2)},
        "score_increase": improvement,
        "improvement_percentage": improvement_pct,
        "added_skills": sorted(skills_v2 - skills_v1),
        "removed_skills": sorted(skills_v1 - skills_v2),
        "better_version": "Resume V2" if score_v2 >= score_v1 else "Resume V1",
        "recommendations": v2.get("recommendations", [])[:8],
    }


def build_local_chat_response(
    user_message: str,
    analysis_result: dict[str, Any],
    resume_text: str = "",
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    profile_data = analysis_result.get("profile", {})
    role = profile_data.get("role", "your detected role")
    industry = profile_data.get("industry", "your industry")
    message = normalize_text(user_message)

    if any(term in message for term in ("weakness", "gap", "low", "missing")):
        gaps = analysis_result.get("gaps", [])[:5]
        return f"Based on your section analysis, for your {role} profile in {industry}, the main weaknesses are: " + format_inline_list(gaps)
    if any(term in message for term in ("strength", "good", "strong")):
        strengths = analysis_result.get("strengths", [])[:5]
        return f"Based on your Overview and section scores, your strongest {role} signals are: " + format_inline_list(strengths)
    if "certification" in message or "certificate" in message:
        certs = profile_data.get("certifications", [])[:5]
        return f"Based on your Certifications and target role context, relevant certifications for {role}: " + format_inline_list(certs)
    if "project" in message or "portfolio" in message:
        improvements = analysis_result.get("personalized_improvements", [])[:4]
        return f"Based on your Projects/Portfolio section, for {role}, focus on role-relevant proof of work: " + format_inline_list(improvements)
    if "job" in message or "fit" in message:
        return f"Based on your detected profile and Skills section, the best-fit direction is {role} in {industry}. Prioritize postings where your visible skills overlap with the matched-skill list."
    if "ats" in message or "score" in message:
        ats = analysis_result.get("ats", {})
        issues = [issue["issue"] for issue in ats.get("issues", [])[:5]]
        return f"Based on your ATS section, your ATS score is {ats.get('score', 0)}/100. Main ATS issues: " + format_inline_list(issues)
    if "learn" in message or "roadmap" in message or "next" in message:
        weeks = analysis_result.get("roadmap", {}).get("weeks", [])
        lines = [f"{week['week']}: {week['focus']}" for week in weeks]
        return f"Based on your Roadmap section, your next learning path should stay within {role}: " + " ".join(lines)

    recommendations = analysis_result.get("recommendations", [])[:5]
    return f"Based on your uploaded resume sections, I will stay within your {role} profile. The best next fixes are: " + format_inline_list(recommendations)


def format_inline_list(items: list[str]) -> str:
    clean = [str(item).strip().rstrip(".") for item in items if str(item).strip()]
    return "; ".join(clean) + ("." if clean else "No clear items were detected yet.")


def domain_safe_list(items: list[str] | tuple[str, ...], profile: RoleProfile) -> list[str]:
    safe: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        if is_avoid_term(text, profile):
            continue
        safe.append(text)
    if not safe and profile.improvements:
        safe = [profile.improvements[0]]
    return safe


def domain_safe_issues(issues: list[dict[str, str]], profile: RoleProfile) -> list[dict[str, str]]:
    safe = []
    for issue in issues:
        combined = " ".join(str(value) for value in issue.values())
        if is_avoid_term(combined, profile):
            continue
        safe.append(issue)
    return safe


def is_avoid_term(text: str, profile: RoleProfile) -> bool:
    if not profile.avoid_terms:
        return False
    normalized = normalize_text(text)
    return any(keyword_in_text(term, normalized) for term in profile.avoid_terms)


def clean_text(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w+#.-]+\b", text or ""))


def keyword_in_text(keyword: str, text: str) -> bool:
    normalized_keyword = normalize_text(keyword)
    normalized_text = normalize_text(text)
    if not normalized_keyword or not normalized_text:
        return False
    parts = [re.escape(part) for part in normalized_keyword.split()]
    pattern = r"[\s_.:/-]+".join(parts)
    return bool(re.search(rf"(?<![a-z0-9+#]){pattern}(?![a-z0-9+#])", normalized_text, flags=re.IGNORECASE))


def count_keyword(keyword: str, text: str) -> int:
    normalized_keyword = normalize_text(keyword)
    if not normalized_keyword:
        return 0
    parts = [re.escape(part) for part in normalized_keyword.split()]
    pattern = r"[\s_.:/-]+".join(parts)
    return len(re.findall(rf"(?<![a-z0-9+#]){pattern}(?![a-z0-9+#])", text, flags=re.IGNORECASE))


def extract_metrics(text: str) -> list[str]:
    patterns = (
        r"\b\d+(?:\.\d+)?[ \t]*(?:%|percent|x|k|m|lpa|crore|lakh|hours|days|weeks|months|years|patients|students|users|customers|clients|projects|cases|roles|hires|contracts|reports)\b",
        r"\b(?:increased|reduced|improved|saved|managed|handled|led|taught|treated|supported|reviewed)\s+[^.\n]{0,60}\d+",
    )
    metrics: list[str] = []
    for pattern in patterns:
        metrics.extend(match.strip() for match in re.findall(pattern, text or "", flags=re.IGNORECASE))
    return list(dict.fromkeys(metrics))


def extract_meaningful_keywords(text: str, limit: int = 20) -> list[str]:
    words = re.findall(r"\b[a-zA-Z][a-zA-Z+#.-]{2,}\b", normalize_text(text))
    stopwords = {
        "and",
        "the",
        "for",
        "with",
        "from",
        "this",
        "that",
        "will",
        "your",
        "our",
        "are",
        "you",
        "have",
        "has",
        "job",
        "role",
        "work",
        "team",
        "candidate",
        "experience",
    }
    counts = Counter(word.strip(".-") for word in words if word not in stopwords)
    return [word for word, _ in counts.most_common(limit)]


def has_contact_info(text: str, contact_details: dict[str, Any] | None = None) -> bool:
    details = contact_details or extract_contact_details(text)
    return details.get("email_status") == "valid" and details.get("phone_status") == "found"


def has_symbol_noise(text: str) -> bool:
    return bool(re.search(r"[\u2600-\u27BF\uE000-\uF8FF]", text or ""))


def heat_level(score: int) -> str:
    if score >= 75:
        return "green"
    if score >= 50:
        return "yellow"
    return "red"


def compact_excerpt(text: str, max_chars: int = 360) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip() + "..."


def clamp(value: int | float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, round(value)))
