from flask_app import db
from page_model import PageModel
import numpy
from cython_tste import cy_tste

db_data = PageModel.query.all()
triplets = []
for pg_model in db_data:
    indices = pg_model.get_index_list()
    if not None in indices:
        triplets.append(indices)
        
triplets = numpy.array(triplets)
print triplets
embedding = cy_tste.tste(triplets,
     no_dims=2,
     lamb=0,
     alpha=None,
     verbose=True,
     max_iter=100,
     save_each_iteration=False,
     initial_X=None,
     static_points=numpy.array([]),
     ignore_zeroindexed_error=True,
     num_threads=None,
     use_log=False,
     )
print embedding