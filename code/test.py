# %%
import os
import argparse
import PyPDF2
from PyPDF2 import PdfFileReader, PdfFileWriter
from PyPDF2.generic import DecodedStreamObject, EncodedStreamObject, NameObject


def replace_text(content, replacements=dict()):
    lines = content.splitlines()

    result = ""
    in_text = False

    for line in lines:
        if line == "BT":
            in_text = True

        elif line == "ET":
            in_text = False

        elif in_text:
            cmd = line[-2:]
            if cmd.lower() == "tj":
                replaced_line = line
                for k, v in replacements.items():
                    replaced_line = replaced_line.replace(k, v)
                result += replaced_line + "\n"
            else:
                result += line + "\n"
            continue

        result += line + "\n"

    return result


def process_data(object, replacements):
    data = object.get_data()
    decoded_data = data.decode("utf-8")

    replaced_data = replace_text(decoded_data, replacements)

    encoded_data = replaced_data.encode("utf-8")
    if object.decoded_self is not None:
        object.decoded_self.set_data(encoded_data)
    else:
        object.set_data(encoded_data)

# %%
in_file = "../manuscript/figure/2/figure2.pdf"
out_decompressed_location = "../manuscript/figure/2/figure2_decompressed.pdf"
out_location = "../manuscript/figure/2/figure2_replaced.pdf"

# %%
!pdftk {in_file} output {out_decompressed_location} uncompress

# %%

# Provide replacements list that you need here
replacements = {"positional_all_scores": "HEYO"}
replacements = [
    ("positional_benchmark_overview", "HEYO")
]

pdf = PyPDF2.PdfReader(open(out_decompressed_location, "rb"))
writer = PyPDF2.PdfWriter() 

for page in pdf.pages:
    contents = page.get_contents().get_data()
    for (a,b) in replacements:
        contents = contents.replace(a.encode('utf-8'), b.encode('utf-8'))
    page.get_contents().set_data(contents)
    writer.add_page(page)
    
with open(out_location, "wb") as f:
     writer.write(f)


    # %%

    # if __name__ == "__main__":
    #     ap = argparse.ArgumentParser()
    #     ap.add_argument("-i", "--input", required=True, help="path to PDF document")
    #     args = vars(ap.parse_args())

    #     in_file = args["input"]
    #     filename_base = in_file.replace(os.path.splitext(in_file)[1], "")

    # Provide replacements list that you need here
    replacements = {"PDF": "DOC"}

    pdf = PdfFileReader(in_file)
    writer = PdfFileWriter()

#     for page_number in range(0, pdf.getNumPages()):

#         page = pdf.getPage(page_number)
#         contents = page.getContents()

#         if isinstance(contents, DecodedStreamObject) or isinstance(
#             contents, EncodedStreamObject
#         ):
#             process_data(contents, replacements)
#         elif len(contents) > 0:
#             for obj in contents:
#                 if isinstance(obj, DecodedStreamObject) or isinstance(
#                     obj, EncodedStreamObject
#                 ):
#                     streamObj = obj.getObject()
#                     process_data(streamObj, replacements)

#         # Force content replacement
#         page[NameObject("/Contents")] = contents.decodedSelf
#         writer.addPage(page)

#     with open(filename_base + ".result.pdf", "wb") as out_file:
#         writer.write(out_file)
