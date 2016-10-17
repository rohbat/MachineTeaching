class PageModel:
    def __init__(self, main_img, compare_img_1, compare_img_2):
        self.main_img = main_img
        self.compare_img_1 = compare_img_1
        self.compare_img_2 = compare_img_2
        self.chosen = None

    def set_chosen(self, img):
        self.chosen = img

    def get_imgs_list(self):
        return [self.main_img, self.compare_img_1, self.compare_img_2]
