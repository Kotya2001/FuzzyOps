# основная папка проекта
# в этом файле - только импорты из подпапок
# и уже отсюда будут делаться импорты в проектах

# так же, чтобы избежать проблем, используйте relative import
# с точкой в начале, иначе будет появляться "module not found"

from .subdir_1 import hello as hello1
from .subdir_2 import hello as hello2
