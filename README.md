#Модель машинного обучения прогнозирования поещаемости предприятия(вуза)
##В данной работе мы проверили возможно ли решить проблему прогнозирования посещаемости студентами занятий, если да,то с какой точностью
###К даной задаче мы подошли как к задаче классификации студентов пропускающих занятия
Обученная модель прогнозирует посещение студента при определенных условиях с точностью 0.93 на данных на которых обучалсь модель
и точность 0.8 на данных, которых модель не видела. Но это не значит, что модель дает исчерпывающий ответ задачу прогнозирования. Остается неопределенность и вероятность возникновения иных событий(признаков), которые могут составлять зависимость, при которой прогноз, может поменяться
##Признаки
* Причина пропуска int64
* День недели int64
* Траты на транспорт int64
* Расстояние от дома до места обучения int64
* Возраст int64
* Количество детей int64
* Выпивает int64
* Курит int64
* Домашние животные int64
## Install
Для запуска проекта необходимо поставить следующие зависимости:
* Установить Python3.6+
Для Windows:
pip install sklearn
pip install IPython
pip install pandas
pip install joblib
pip install graphviz
Для Linux:
pip3 install sklearn
pip3 install IPython
pip3 install pandas
pip3 install joblib
pip3 install graphviz
### Запуск
Для Windows:
python -m ./paht_dataset.csv ./paht_test.csv
Для Linux:
python3.8 -m ./paht_dataset.csv ./paht_test.csv
###Разработали модель
Третьяк Михаил, Фуников Владислав
###Оформлением отчета занимались
Щекатурин Александр, Орехов Павел
