// get_preview.js
require('dotenv').config();
const spotifyPreviewFinder = require('spotify-preview-finder');

async function getPreview() {
  const query = process.argv.slice(2).join(' ');
  if (!query) {
    console.error("No query provided.");
    process.exit(1);
  }
  
  try {
    // Search for one result.
    const result = await spotifyPreviewFinder(query, 1);
    if (result.success && result.results && result.results.length > 0) {
      // Output only the first preview URL, without any additional text.
      const previewUrl = result.results[0].previewUrls[0];
      if (previewUrl) {
        console.log(previewUrl);
      } else {
        console.error("No preview URL found.");
        process.exit(1);
      }
    } else {
      console.error("No preview URL found:", result.error || "Unknown error");
      process.exit(1);
    }
  } catch (error) {
    console.error("Error:", error.message);
    process.exit(1);
  }
}

getPreview();
