# Install Python packages
echo "Installing Hecate Python packages"

cd $HECATE/python/hecate
pip uninstall -y hecate 2>/dev/null || true
pip install -e . 

cd $HECATE/python/poly
pip uninstall -y poly 2>/dev/null || true
pip install -e . 

cd $HECATE/python/hetorch
pip uninstall -y hetorch 2>/dev/null || true
pip install -e .


