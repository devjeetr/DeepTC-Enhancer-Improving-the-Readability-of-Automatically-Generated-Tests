/**
 * 1. Creates a new KeyCloakUriBuilder "uri" from path
 * 2. Creates a new  HashMap and uses it to create a new URI "result" using
 *    method "buildFromEncodedMap" of "uri"
 * 3. Checks if the raw path of "result" equals "x"
 */
@Test(timeout = 4000)
public void testEncodedPath()  throws Throwable  {
    KeycloakUriBuilder uri = KeycloakUriBuilder.fromPath("x");
    HashMap<String, Integer> map = new HashMap<String, Integer>();
    URI result = uri.buildFromEncodedMap(map);
    assertEquals("x", result.getRawPath());
}