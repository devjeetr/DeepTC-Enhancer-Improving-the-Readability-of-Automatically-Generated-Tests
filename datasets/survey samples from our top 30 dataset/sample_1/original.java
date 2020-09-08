
/**
1. Creates a new KeyCloakUriBuilder using aFTPURI of MockURI,
    and checks if its port is -1 and host is not null.
*/
@Test(timeout = 4000)
public void test029()  throws Throwable  {
    URI uRI0 = MockURI.aFTPURI;
    KeycloakUriBuilder keycloakUriBuilder0 = KeycloakUriBuilder.fromUri(uRI0);
    String string0 = keycloakUriBuilder0.getHost();
    assertEquals((-1), keycloakUriBuilder0.getPort());
    assertNotNull(string0);
}