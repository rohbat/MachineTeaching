from flask_app_crowd import db
from page_model_bird import PageModel
import numpy
from cython_tste import cy_tste
import cPickle
import random

random.seed(1)
numpy.random.seed(1)
db_data = PageModel.query.all()
#f = open("triplets_seabed.txt", "rb")
#db_data = cPickle.load(f)
#f.close()
print len(db_data)
triplets = []
for pg_model in db_data:
    indices = pg_model.get_index_list()
    if not None in indices:
        triplets.append(indices)
        
triplets = numpy.array(triplets)
print triplets
embedding = cy_tste.tste(triplets,
     no_dims=5,
     lamb=0,
     alpha=None,
     verbose=True,
     max_iter=4000,
     save_each_iteration=False,
     initial_X=None,
     static_points=numpy.array([]),
     ignore_zeroindexed_error=True,
     num_threads=None,
     use_log=False,
     )
print embedding
numpy.save("X_initial_bird", embedding)
embedding1 = numpy.load("X_initial_bird.npy")
print embedding1