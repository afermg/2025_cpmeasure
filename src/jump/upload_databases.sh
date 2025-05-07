# Find the latest version of the dataset
ZENODO_ENDPOINT="https://zenodo.org"
DEPOSITION_PREFIX="${ZENODO_ENDPOINT}/api/deposit/depositions"
ORIGINAL_ID="15359151"
DIR_TO_VERSION="$1"

if [ -z "${ORIGINAL_ID}" ]; then # Only get latest id when provided an original one
	echo "Creating new deposition"
	DEPOSITION_ENDPOINT="${DEPOSITION_PREFIX}"
else # Update existing dataset
	echo "Creating new version"
	LATEST_ID=$(curl "${ZENODO_ENDPOINT}/records/${ORIGINAL_ID}/latest" |
		grep records | sed 's/.*href=".*\.org\/records\/\(.*\)".*/\1/')
	DEPOSITION_ENDPOINT="${DEPOSITION_PREFIX}/${LATEST_ID}/actions/newversion"
fi

if [ -z "${ZENODO_TOKEN}" ]; then # Check Zenodo Token
	echo "Access token not available"
	exit 1
else
	echo "Access token found."
fi

# Create new deposition
DEPOSITION=$(curl -H "Content-Type: application/json" \
	-X POST --data "{}" \
	"${DEPOSITION_ENDPOINT}?access_token=${ZENODO_TOKEN}" |
	jq .id)
echo "New deposition ID is ${DEPOSITION}"

# Variables
curl "${DEPOSITION_PREFIX}?access_token=${ZENODO_TOKEN}"
BUCKET_DATA=$(curl "${DEPOSITION_PREFIX}/${DEPOSITION}?access_token=${ZENODO_TOKEN}")
BUCKET=$(echo "${BUCKET_DATA}" | jq --raw-output .links.bucket)

if [ "${BUCKET}" = "null" ]; then
	echo "Could not find URL for upload. Response from server:"
	echo "${BUCKET_DATA}"
	exit 1
fi

# Upload file
echo "Uploading files to bucket ${BUCKET}"
for FILE_TO_VERSION in $(find "${DIR_TO_VERSION}" -name '*'); do
	echo "${FILE_TO_VERSION}"
	curl --retry 5 \
		--retry-delay 5 \
		-o /dev/null \
		--upload-file ${FILE_TO_VERSION} \
		"${BUCKET}/${FILE_TO_VERSION##*/}?access_token=${ZENODO_TOKEN}"
done

# Upload Metadata
echo -e '{"metadata": {
    "title": Subset of JUMP images with distinctive perturbations" ,
    "creators": [
        {
            "name": "Alán F. Muñoz"
        }
    ],
"description":"<p>This dataset provides a subset of JUMP focusing on perturbations that showcase the most distinctiveness for every feature in JUMP. It contains the originals election manually curated from&nbsp;<a href="https://github.com/broadinstitute/monorepo/tree/main/libs/jump_rr#quick-data-access">JUMP_rr tables</a>, a parquet for convenience with the index, and a compressed tarball with all the tiff files.</p>",
"upload_type": "dataset",
"access_right": "open"
}}' >metadata.json

NEW_DEPOSITION_ENDPOINT="${DEPOSITION_PREFIX}/${DEPOSITION}"
echo "Uploading file to ${NEW_DEPOSITION_ENDPOINT}"
curl -H "Content-Type: application/json" \
	-X PUT --data @metadata.json \
	"${NEW_DEPOSITION_ENDPOINT}?access_token=${ZENODO_TOKEN}"

# Publish
# echo "Publishing to ${NEW_DEPOSITION_ENDPOINT}"
curl -H "Content-Type: application/json" \
	-X POST --data "{}" \
	"${NEW_DEPOSITION_ENDPOINT}/actions/publish?access_token=${ZENODO_TOKEN}" |
	jq .id
