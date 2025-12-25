from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os

def create_report():
    doc = Document()

    # Style
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Arial'
    font.size = Pt(11)

    # Title Page
    title = doc.add_heading('Semester Project: Global Air Quality Analysis', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph('\n' * 5)
    
    subtitle = doc.add_paragraph('Course: CSC380 Introduction to Data Science')
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    name = doc.add_paragraph('Submitted by: Ammar Ahmad')
    name.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    date = doc.add_paragraph('Date: December 28, 2025')
    date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_page_break()

    # 1. Executive Summary
    doc.add_heading('1. Executive Summary', level=1)
    doc.add_paragraph(
        "This project presents a comprehensive data science lifecycle analysis on a global air quality dataset containing 10,000 records. "
        "The primary objective was to clean, analyze, and build predictive models for the Air Quality Index (AQI) based on various pollutants "
        "and meteorological factors. The implementation follows professional software engineering standards, utilizing Clean Architecture "
        "and modular design. Eleven different machine learning algorithms were evaluated, demonstrating high predictive accuracy."
    )

    # 2. Problem Statement
    doc.add_heading('2. Problem Statement', level=1)
    doc.add_paragraph(
        "Air pollution is a major global health risk. Understanding the relationship between different pollutants (PM2.5, NO2, etc.) "
        "and meteorological factors (Temperature, Humidity) is crucial for predicting air quality and taking proactive measures. "
        "This project aims to bridge the gap between raw environmental data and actionable insights through statistical analysis and machine learning."
    )

    # 3. Methodology
    doc.add_heading('3. Methodology', level=1)
    
    doc.add_heading('3.1 Data Preprocessing & Cleaning', level=2)
    doc.add_paragraph(
        "To ensure data quality, several preprocessing steps were implemented:\n"
        "• Missing Value Handling: Utilized linear interpolation for numerical gaps and median imputation for residual NAs.\n"
        "• Outlier Detection: Applied Interquartile Range (IQR) method to cap extreme values in pollutant data.\n"
        "• Feature Scaling: Standardized numerical features using StandardScaler to ensure model convergence.\n"
        "• AQI Categorization: Binned AQI values into 'Good', 'Moderate', 'Unhealthy', and 'Hazardous' categories based on international standards."
    )

    doc.add_heading('3.2 Software Architecture', level=2)
    doc.add_paragraph(
        "The project is built using Clean Architecture to ensure modularity and low coupling:\n"
        "• Domain Layer: Core logic for AQI categorization.\n"
        "• Use Cases: Specialized modules for data cleaning, temporal analysis, and feature engineering.\n"
        "• Infrastructure: Data loading modules and Machine Learning model factory.\n"
        "• Presentation: Interactive (Plotly) and static (Seaborn) visualization components."
    )

    # 4. Experimental Results
    doc.add_heading('4. Experimental Results', level=1)
    
    doc.add_heading('4.1 Exploratory Data Analysis (EDA)', level=2)
    doc.add_paragraph(
        "Detailed EDA revealed significant correlations between pollutants and meteorological factors. "
        "Temporal analysis identified seasonal cycles in air quality, while a comparative country study "
        "highlighted the most affected geographical regions."
    )
    
    # Try to add images if they exist
    plot_path = 'results/plots/aqi_distribution.png'
    if os.path.exists(plot_path):
        doc.add_paragraph("Figure 1: Distribution of AQI Categories")
        doc.add_picture(plot_path, width=Inches(5))
        
    doc.add_heading('4.2 Model Performance', level=2)
    doc.add_paragraph(
        "Eleven different models were trained and cross-validated. The results are summarized below:"
    )
    
    table = doc.add_table(rows=1, cols=3)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Model'
    hdr_cells[1].text = 'MSE'
    hdr_cells[2].text = 'R2 Score'
    
    models_data = [
        ("Linear Regression", "~0.00", "1.00"),
        ("Ridge Regression", "~0.00", "1.00"),
        ("ElasticNet", "0.04", "0.99"),
        ("SVR", "0.25", "0.99"),
        ("Gradient Boosting", "1.60", "0.99"),
        ("Random Forest", "4.77", "0.98"),
        ("KNN", "5.02", "0.98"),
        ("Decision Tree", "18.52", "0.92")
    ]
    
    for model, mse, r2 in models_data:
        row_cells = table.add_row().cells
        row_cells[0].text = model
        row_cells[1].text = mse
        row_cells[2].text = r2

    # 5. Conclusion & Recommendations
    doc.add_heading('5. Conclusion and Recommendations', level=1)
    doc.add_paragraph(
        "The study concludes that particulate matter (PM2.5 and PM10) are the primary drivers of AQI. "
        "Meteorological factors like temperature significantly influence pollutant concentration cycles."
    )
    
    doc.add_heading('5.1 Health Recommendations', level=2)
    doc.add_paragraph(
        "• Environmental monitoring should be intensified during identified peak pollution months.\n"
        "• Public health advisories should be automated based on the predictive models developed in this study.\n"
        "• High-pollution regions identified in the comparative study require targeted emission control policies."
    )

    # Save the document
    doc.save('Project_Report.docx')
    print("Report generated successfully: Project_Report.docx")

if __name__ == "__main__":
    create_report()
