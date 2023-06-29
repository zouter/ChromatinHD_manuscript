from manuscript import Manuscript as ManuscriptBase


class Manuscript(ManuscriptBase):
    def _replace_texts(self, text):
        text = text.replace("ChromatinHD predictive", "ChromatinHD predictive")
        text = text.replace("ChromatinHD differential", "ChromatinHD differential")
        text = text.replace("ChromatinHD differential", "ChromatinHD differential")
        return text
