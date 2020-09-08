/**
 * 1. Creates a new KeyCloakUriBuilder "keycloakUriBuilder0" from path.
 * 2. Creates a new  HashMap and uses it to create a new URI "uRI0" using
 *    method "buildFromEncodedMap" of "keycloakUriBuilder0"
 * 3. Checks if the raw path of "uRI0" equals "x"
 */
@Test(timeout = 4000)
public void test040()  throws Throwable  {
    KeycloakUriBuilder keycloakUriBuilder0 = KeycloakUriBuilder.fromPath("x");
    HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
    URI uRI0 = keycloakUriBuilder0.buildFromEncodedMap(hashMap0);
    assertEquals("x", uRI0.getRawPath());
}