lint:
	python -m flake8 --ignore W3,E3,E5,E74 rankfm/

test:
	python -m pytest -r Efp tests/