from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def demo_hello_world():
    print("Running the demo pdf function")
    # create a Canvas object with a filename
    c = canvas.Canvas(r"C:\Users\gac8\PycharmProjects\PS-PDF\pdf\outputs\rl-hello_again.pdf", pagesize=A4)  # A4 pagesize
    # draw a string at x=100, y=800 points
    # point ~ standard desktop publishing (72 DPI)
    # coordinate system:
    #   y
    #   |
    #   |   page
    #   |
    #   |
    #   0-------x
    c.drawString(50, 780, "Hello Again")
    # finish page
    c.showPage()
    # construct and save file to .pdf
    c.save()