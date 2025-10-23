from fpdf import FPDF
import os

def convert_txt_to_pdf(input_file, output_file):
    # Create PDF object
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    
    # Read text file
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if paragraph.strip():
                # Write paragraph
                pdf.multi_cell(0, 8, text=paragraph.strip())
                # Add some space between paragraphs
                pdf.ln(4)
    
    # Save PDF
    pdf.output(output_file)

def main():
    # Convert all txt files in data directory to PDFs
    data_dir = 'data'
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            txt_path = os.path.join(data_dir, filename)
            pdf_path = os.path.join(data_dir, filename.replace('.txt', '.pdf'))
            convert_txt_to_pdf(txt_path, pdf_path)
            print(f"Converted {filename} to PDF")

if __name__ == "__main__":
    main()