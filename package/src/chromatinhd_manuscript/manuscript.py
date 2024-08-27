from manuscript import Manuscript as ManuscriptBase


class Manuscript(ManuscriptBase):
    def _replace_texts(self, text):
        text = text.replace("ChromatinHD-pred", r"\textit{ChromatinHD-pred}")
        text = text.replace("ChromatinHD-diff", r"\textit{ChromatinHD-diff}")
        text = super()._replace_texts(text)
        return text
