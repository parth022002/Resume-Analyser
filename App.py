import streamlit as st
import nltk
import random
import spacy
import pandas as pd
import base64
import time, datetime
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io
from streamlit_tags import st_tags
from PIL import Image
import pymysql
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
import pafy
import plotly.express as px
from knn_algorithm import predict_category

nltk.download('stopwords')
spacy.load('en_core_web_sm')

# Database Connection
def init_connection():
    return pymysql.connect(host='localhost', user='root', password='root', db='sra')

connection = init_connection()
cursor = connection.cursor()

# Utility functions
def fetch_yt_video(link):
    video = pafy.new(link)
    return video.title

def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
    
    text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def course_recommender(course_list):
    st.subheader("**Courses & Certificates🎓 Recommendations**")
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 4)
    random.shuffle(course_list)
    for c in range(no_of_reco):
        c_name, c_link = course_list[c]
        st.markdown(f"({c+1}) [{c_name}]({c_link})")
        rec_course.append(c_name)
    return rec_course

def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills, courses):
    DB_table_name = 'user_data'
    insert_sql = f"INSERT INTO {DB_table_name} (ID, Name, Email_ID, resume_score, Timestamp, Page_no, Predicted_Field, User_level, Actual_skills, Recommended_skills, Recommended_courses) VALUES (0, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    rec_values = (name, email, str(res_score), timestamp, str(no_of_pages), reco_field, cand_level, skills, recommended_skills, courses)
    cursor.execute(insert_sql, rec_values)
    connection.commit()

# Streamlit UI
def run():
    st.set_page_config(page_title="Resume Analyzer", page_icon='./Logo/logo.ico')
    st.title("Smart Resume Analyser")
    st.sidebar.markdown("# Choose User")
    activities = ["Normal User", "Admin"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    
    img = Image.open('./Logo/logo.png')
    img = img.resize((250, 250))
    st.image(img)

    if choice == 'Normal User':
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        if pdf_file is not None:
            save_image_path = f'./Uploaded_Resumes/{pdf_file.name}'
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)

            resume_data = ResumeParser(save_image_path).get_extracted_data()
            if resume_data:
                resume_text = pdf_reader(save_image_path)
                st.header("**Resume Analysis**")
                st.success(f"Hello {resume_data['name']}")
                st.subheader("**Your Basic info**")
                st.text(f"Name: {resume_data.get('name', 'N/A')}")
                st.text(f"Email: {resume_data.get('email', 'N/A')}")
                st.text(f"Contact: {resume_data.get('mobile_number', 'N/A')}")
                st.text(f"Resume pages: {resume_data.get('no_of_pages', 'N/A')}")

                cand_level = ''
                pages = resume_data.get('no_of_pages', 0)
                if pages == 1:
                    cand_level = "Fresher"
                    st.markdown('<h4 style="color: #d73b5c;">You are looking Fresher.</h4>', unsafe_allow_html=True)
                elif pages == 2:
                    cand_level = "Intermediate"
                    st.markdown('<h4 style="color: #1ed760;">You are at intermediate level!</h4>', unsafe_allow_html=True)
                elif pages >= 3:
                    cand_level = "Experienced"
                    st.markdown('<h4 style="color: #fba171;">You are at experience level!</h4>', unsafe_allow_html=True)

                st.subheader("**Skills Recommendation💡**")
                keywords = st_tags(label='### Skills that you have', text='See our skills recommendation', value=resume_data.get('skills', []), key='1')

                recommended_skills = []
                reco_field = predict_category(" ".join(resume_data.get('skills', [])))
                st.success(f"**Our analysis says you are looking for {reco_field} Jobs.**")

                if reco_field == 'Data Science':
                    recommended_skills = ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling', 'Data Mining', 'Clustering & Classification', 'Data Analytics', 'Quantitative Analysis', 'Web Scraping', 'ML Algorithms', 'Keras', 'Pytorch', 'Probability', 'Scikit-learn', 'Tensorflow', "Flask", 'Streamlit']
                elif reco_field == 'Web Development':
                    recommended_skills = ['React', 'Django', 'Node JS', 'React JS', 'php', 'laravel', 'Magento', 'wordpress', 'Javascript', 'Angular JS', 'c#', 'Flask', 'SDK']
                elif reco_field == 'Android Development':
                    recommended_skills = ['Android', 'Android Development', 'Flutter', 'Kotlin', 'XML', 'Java', 'Android Studio', 'SDK', 'React Native']
                elif reco_field == 'IOS Development':
                    recommended_skills = ['IOS', 'IOS Development', 'Swift', 'Cocoa', 'Xcode', 'Objective-C']
                elif reco_field == 'UI-UX Development':
                    recommended_skills = ['UI/UX', 'CSS', 'Sketch', 'Adobe XD', 'Figma', 'Zeplin', 'Balsamiq', 'Wireframes', 'Prototyping', 'Storyframes', 'Adobe Photoshop', 'Editing', 'Adobe Illustrator', 'Illustrator', 'Adobe After Effects', 'After Effects', 'Adobe Premier Pro', 'Premier Pro', 'Adobe Indesign', 'Indesign', 'Wireframe', 'Solid', 'Grasp', 'User Research']
                
                st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System', value=recommended_skills, key='2')

                rec_course = course_recommender(ds_course if reco_field == 'Data Science' else web_course if reco_field == 'Web Development' else android_course if reco_field == 'Android Development' else ios_course if reco_field == 'IOS Development' else uiux_course)

                ts = time.time()
                timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                insert_data(resume_data['name'], resume_data['email'], resume_data['no_of_pages']*20, timestamp, resume_data['no_of_pages'], reco_field, cand_level, resume_data['skills'], recommended_skills, rec_course)

    elif choice == 'Admin':
        st.success('Welcome to Admin Section')
        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')
        if st.button('Login'):
            if ad_user == 'admin' and ad_password == 'admin123':
                st.success("Welcome Admin")
                cursor.execute('SELECT * FROM user_data')
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Page no', 'Predicted Field', 'User Level', 'Actual Skills', 'Recommended Skills', 'Recommended Courses'])
                st.dataframe(df)
                st.markdown(get_table_download_link(df, 'User_Data.csv', 'Download Report'), unsafe_allow_html=True)
            else:
                st.error("Wrong ID & Password Provided")

if __name__ == '__main__':
    run()
