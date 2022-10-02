install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt

venv:
	test -d venv || python3 -m venv venv

run:
	@venv/bin/python3 src/wrapper.py

clean:
	rm -rf venv
	find -iname "*.pyc" -delete