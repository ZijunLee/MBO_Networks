#import three_moons
#reload(three_moons)
#import read_mnist
#reload(read_mnist)
#from .three_moons import three_moons
#from .read_mnist import read_mnist
#from .read_mnist import subsample

from os.path import dirname, basename, isfile, join
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]