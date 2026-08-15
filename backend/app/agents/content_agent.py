import urllib.parse
from typing import Dict, Any

class ContentAgent:
    """
    CV & Application Package Generator with Overleaf Integration.
    """
    
    @staticmethod
    def generate_application_package(job_title: str, company: str, candidate_name: str = "Arjun B.", template_id: str = "jakes-resume") -> Dict[str, Any]:
        
        # Sample LaTeX Resume Source
        latex_template = f"""\\documentclass[letterpaper,11pt]{{article}}
\\usepackage{{latexsym}}
\\usepackage[empty]{{fullpage}}
\\usepackage{{titlesec}}
\\usepackage{{marvosym}}
\\usepackage[usenames,dvipsnames]{{color}}
\\usepackage{{enumitem}}
\\usepackage[hidelinks]{{hyperref}}
\\usepackage{{fancyhdr}}
\\pagestyle{{fancy}}
\\fancyhf{{}} 
\\renewcommand{{\\headrulewidth}}{{0pt}}

\\begin{{document}}
\\begin{{center}}
    {{\\Huge \\scshape {candidate_name}}} \\\\ \\vspace{{1pt}}
    \\small Bengaluru, India $|$ \\href{{mailto:arjun.b@talentforge.ai}}{{arjun.b@talentforge.ai}} $|$ \\href{{https://linkedin.com/in/arjun-b}}{{linkedin.com/in/arjun-b}} $|$ \\href{{https://github.com/arjun-b}}{{github.com/arjun-b}}
\\end{{center}}

\\section{{Target Role}}
Tailored for \\textbf{{{job_title}}} at \\textbf{{{company}}}.

\\section{{Education}}
  \\resumeSubHeadingListStart
    \\resumeSubheading
      {{Indian Institute of Technology / B.Tech Computer Science}}{{Bengaluru, KA}}
      {{Bachelor of Technology in Computer Science; GPA: 8.9/10.0}}{{2021 -- 2025}}
  \\resumeSubHeadingListEnd

\\section{{Technical Skills}}
 \\begin{{itemize}}[leftmargin=0.15in, label={{}}]
    \\small{{\\item{{
     \\textbf{{Languages}}{{: Python, Go, SQL, JavaScript, HTML/CSS}} \\\\
     \\textbf{{Frameworks}}{{: FastAPI, React, Node.js, Flask, PyTorch}} \\\\
     \\textbf{{Developer Tools}}{{: Docker, Kubernetes, AWS, Git, PostgreSQL, Redis, Kafka}}
    }}}}
 \\end{{itemize}}

\\end{{document}}
"""
        
        # Overleaf URL builder using standard snip parameter
        encoded_latex = urllib.parse.quote(latex_template)
        # Using data URI snippet link
        overleaf_url = f"https://www.overleaf.com/docs?snip={encoded_latex}"
        
        cover_letter = f"""Dear Hiring Manager at {company},

I am writing to express my enthusiastic interest in the {job_title} role. With a strong background in building high-throughput backend microservices using Python, FastAPI, and AWS, I have delivered resilient cloud solutions that serve millions of API requests efficiently.

At Superset Inc., I led the implementation of asynchronous event processing pipelines reducing latency by 40%. My technical expertise matches your team's requirements for scalable distributed systems.

I welcome the opportunity to discuss how my skill set and passion for software craftsmanship can contribute to {company}'s ongoing success.

Sincerely,
{candidate_name}"""

        qa_answers = {
            "why_role": f"I am deeply inspired by {company}'s engineering culture and mission. The {job_title} role directly aligns with my experience in scaling cloud backend services.",
            "notice_period": "30 days (negotiable)",
            "salary_expectation": "₹ 18 - 22 LPA (commensurate with role and market standards)"
        }
        
        return {
            "template_used": template_id,
            "latex_code": latex_template,
            "overleaf_url": overleaf_url,
            "cover_letter": cover_letter,
            "qa_answers": qa_answers,
            "package_assembled": True
        }

content_agent = ContentAgent()
