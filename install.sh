# Install Python packages
echo "Installing Hecate Python packages"

cd $HECATE/python/hecate
python setup.py sdist --format=tar
pip uninstall -y hecate 2>/dev/null || true
pip install dist/hecate-0.0.1.tar

cd $HECATE/python/poly
python setup.py sdist --format=tar
pip uninstall -y poly 2>/dev/null || true
pip install dist/poly-0.0.1.tar

cd $HECATE/python/hetorch
pip uninstall -y hetorch 2>/dev/null || true
pip install -e .


