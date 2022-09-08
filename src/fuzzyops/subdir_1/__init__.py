# переносим весь код из файлов нижнего уровня в этот,
# чтобы отсюда перенести импорты в сам проект
from .package_1 import hello_from_subdir as hello
