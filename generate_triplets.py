from flask_app import db
from page_model import PageModel
import numpy

db_data = PageModel.query.all()
triplets = []
for pg_model in db_data:
    indices = pg_model.get_index_list()
    if not None in indices:
        triplets.append(indices)
triplets = numpy.array(triplets)
print triplets