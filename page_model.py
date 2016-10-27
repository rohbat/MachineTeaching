from sqlalchemy import Column, Integer, String
from database import db

class PageModel(db.Model):
    __tablename__ = 'triplets'
    id = Column(Integer, primary_key=True)
    main_img = Column(String(100))
    compare_img_1 = Column(String(100))
    compare_img_2 = Column(String(100))
    chosen = Column(String)

    def __init__(self, main_img, compare_img_1, compare_img_2):
        self.main_img = main_img
        self.compare_img_1 = compare_img_1
        self.compare_img_2 = compare_img_2
        self.chosen = ''

    def set_chosen(self, img):
        self.chosen = img

    def get_imgs_list(self):
        return [self.main_img, self.compare_img_1, self.compare_img_2]
