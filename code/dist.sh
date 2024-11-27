version="1.0.2"

git add .
git commit -m "version v${version}"

git tag -a v${version} -m "v${version}"

# optional: upload test
# twine upload --repository testpypi dist/eyck-${version}.tar.gz --verbose

git push --tags
git push

# create zip
zip -r dist/chromatinhd_manuscript-${version}.zip . -x "./dist/*" -x "./output/*" -x "./manuscript/*" -x "./results/*"  -x "./tmp/*" -x "./code/1-preprocessing/bulk_datasets/*"  -x "**/*.egg-info/*"   -x ".git/*"  -x ".vscode/*"  -x "**/__old/*"

# conda install gh --channel conda-forge
gh release create v${version} -t "v${version}" -n "v${version}" dist/chromatinhd_manuscript-${version}.zip
