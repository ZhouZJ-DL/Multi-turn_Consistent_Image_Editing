document.addEventListener('DOMContentLoaded', function() {
    const toggleButton = document.getElementById('toggleButton');
    const imageRow = document.getElementById('imageRowContainer');

    if (toggleButton && imageRow) {
        // Check initial state (if row is hidden initially, mark button as collapsed)
        if (imageRow.classList.contains('hidden')) {
             toggleButton.classList.add('collapsed');
             toggleButton.setAttribute('aria-expanded', 'false');
        } else {
             toggleButton.setAttribute('aria-expanded', 'true');
        }


        toggleButton.addEventListener('click', function() {
            // Check if the image row is currently hidden
            const isHidden = imageRow.classList.contains('hidden');

            if (isHidden) {
                // SHOW the row
                imageRow.classList.remove('hidden');
                // Update button state: remove 'collapsed' class (points down)
                toggleButton.classList.remove('collapsed');
                // Update ARIA attribute
                toggleButton.setAttribute('aria-expanded', 'true');
            } else {
                // HIDE the row
                imageRow.classList.add('hidden');
                // Update button state: add 'collapsed' class (points right)
                toggleButton.classList.add('collapsed');
                // Update ARIA attribute
                toggleButton.setAttribute('aria-expanded', 'false');
            }
        });
    } else {
        console.error("Could not find the toggle button or the image row container.");
    }
});