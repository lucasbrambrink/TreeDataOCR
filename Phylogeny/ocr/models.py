from django.db import models
from .utils import PolygonAnalyzer, PILHandler


class ImageAnalysis(models.Model):
    data_source = models.CharField(max_length=100)
    image = models.ImageField()
    processed_polygon = models.TextField()

    def process_polygon(self, commit=False):
        pa = PolygonAnalyzer(self.data_source)
        tree = pa.process_tree_polygon()
        self.processed_polygon = u'&'.join(u'{},{}'.format(*map(str, p))
                                           for p in tree)

        if commit:
            self.save()

    @property
    def processed_pixel_map(self):
        pixels = set()
        for pixel in self.processed_polygon.split(u'&'):
            x, y = pixel.split(u',')
            pixels.add((int(x), int(y)))
        return pixels

    def show(self):
        pil = PILHandler()
        img = pil.create_img_from_pixel_map(self.processed_pixel_map)
        img.show()