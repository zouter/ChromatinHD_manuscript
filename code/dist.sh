version="1.0.0"

git add .
git commit -m "version v${version}"

git tag -a v${version} -m "v${version}"

# optional: upload test
# twine upload --repository testpypi dist/eyck-${version}.tar.gz --verbose

git push --tags
git push

# create zip
zip -r dist/chromatinhd_manuscript-${version}.tar.gz ./ -x dist/

# conda install gh --channel conda-forge
gh release create v${version} -t "v${version}" -n "v${version}" dist/chromatinhd_manuscript-${version}.tar.gz
