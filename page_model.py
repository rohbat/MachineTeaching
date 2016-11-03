from sqlalchemy import Column, Integer, String
from database import db

class PageModel(db.Model):
    __tablename__ = 'triplets'
    id = Column(Integer, primary_key=True)
    main_img = Column(Integer)
    compare_img_1 = Column(Integer)
    compare_img_2 = Column(Integer)
    chosen = Column(Integer)

    def __init__(self, main_img=None, compare_img_1=None, compare_img_2=None, 
                main_path=None, compare_1_path=None, compare_2_path=None):
        self.main_img = main_img
        self.compare_img_1 = compare_img_1
        self.compare_img_2 = compare_img_2

        self.main_path = main_path
        self.compare_1_path = compare_1_path
        self.compare_2_path = compare_2_path

        self.chosen = ''

    def set_chosen(self, img):
        self.chosen = img

    def get_imgs_list(self):
        return [self.main_path, self.compare_1_path, self.compare_2_path]
    
    def get_index_list(self):
        not_chosen = self.compare_img_1
        if self.chosen == not_chosen:
            not_chosen = self.compare_img_2
        return [self.main_img, self.chosen, not_chosen]
