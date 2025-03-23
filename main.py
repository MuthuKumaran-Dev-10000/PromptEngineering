# main.py
import re
import PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from ollama import Client

class ResumeEnhancer:
    def __init__(self, resume_path, jd_path):
        self.client = Client(host='http://localhost:11434')
        self.resume_path = resume_path
        self.jd_path = jd_path
        self.job_description = self._read_jd()
        self.resume_text = self._parse_resume()

    def _parse_resume(self):
        try:
            with open(self.resume_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return '\n'.join([page.extract_text() for page in reader.pages])
        except Exception as e:
            print(f"Error parsing resume: {str(e)}")
            return ''

    def _read_jd(self):
        with open(self.jd_path, 'r') as f:
            return f.read()

    def _llm_enhance(self, prompt):
        response = self.client.generate(model='tinyllama', prompt=prompt)
        return response['response'].strip()

    def analyze_requirements(self):
        prompt = f"""Analyze this job description and extract key technical skills:
        {self.job_description}
        List only the technical skills as comma-separated values:"""
        return self._llm_enhance(prompt).split(',')

    def enhance_resume_section(self, section_name, current_content):
        prompt = f"""Enhance this resume {section_name} section to better match these skills: {', '.join(self.analyze_requirements())}.
        Current content: {current_content}
        Improved content:"""
        return self._llm_enhance(prompt)

    def generate_pdf(self, output_path):
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Enhance Skills Section
        skills_match = re.search(r"(?i)(skills|technical skills)[:\n](.*?)(?=\n\n)", 
                               self.resume_text, re.DOTALL)
        enhanced_skills = self.enhance_resume_section('Skills', skills_match.group(2) if skills_match else "")
        
        # Enhance Experience Section
        exp_match = re.search(r"(?i)(experience|work history)[:\n](.*?)(?=\n\n)", 
                            self.resume_text, re.DOTALL)
        enhanced_exp = self.enhance_resume_section('Experience', exp_match.group(2) if exp_match else "")
        
        story.append(Paragraph("Technical Skills", styles['Heading2']))
        story.append(Paragraph(enhanced_skills, styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Professional Experience", styles['Heading2']))
        story.append(Paragraph(enhanced_exp, styles['BodyText']))
        
        doc.build(story)

if __name__ == "__main__":
    enhancer = ResumeEnhancer("Resume.pdf", "Req.txt")
    enhancer.generate_pdf("NewResume.pdf")
    print("Enhanced resume generated as NewResume.pdf")