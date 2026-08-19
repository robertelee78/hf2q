use super::metadata::{lock_metadata_state, MetadataStateAuthorization};
use super::test_fixture::Fixture;
use super::{
    open_existing_installation_identity, ExplicitRootAuthorization, FirstActivationPreparation,
    LiveInstalledReleaseFloor,
};

fn authorization(fixture: &Fixture) -> MetadataStateAuthorization {
    let identity = open_existing_installation_identity(
        ExplicitRootAuthorization::new(&fixture.root).expect("explicit fixture root"),
    )
    .expect("open installation identity")
    .expect("installation identity exists");
    MetadataStateAuthorization::from_identity(identity)
}

#[test]
fn live_installed_release_floor_is_absent_then_exactly_bound_to_current() {
    let fixture = Fixture::new();
    let authorization = authorization(&fixture);
    let locked = lock_metadata_state(&authorization).expect("lock initial metadata state");
    assert!(matches!(
        locked
            .read_live_installed_release_floor()
            .expect("read absent floor"),
        LiveInstalledReleaseFloor::Absent
    ));
    drop(locked);

    let FirstActivationPreparation::Ready(prepared) =
        fixture.prepare().expect("prepare first activation")
    else {
        panic!("fresh fixture cannot already be activated")
    };
    prepared.commit().expect("commit first activation");

    let locked = lock_metadata_state(&authorization).expect("lock activated metadata state");
    let LiveInstalledReleaseFloor::Active(active) = locked
        .read_live_installed_release_floor()
        .expect("read active floor")
    else {
        panic!("committed current must establish a live floor")
    };
    assert_eq!(active.version().as_str(), "0.2.0");
    assert_eq!(active.target().as_str(), "aarch64-apple-darwin");
    assert_eq!(active.installation_sequence(), 1);
    assert_eq!(active.activation_sequence(), 1);
    assert_ne!(active.receipt_sha256(), [0; 32]);
}

#[test]
fn live_installed_release_floor_rejects_noncanonical_active_state() {
    enum Mutation {
        WrongCurrent,
        ExtraActivation,
        MutatedReceipt,
    }

    for mutation in [
        Mutation::WrongCurrent,
        Mutation::ExtraActivation,
        Mutation::MutatedReceipt,
    ] {
        let fixture = Fixture::new();
        let authorization = authorization(&fixture);
        let FirstActivationPreparation::Ready(prepared) =
            fixture.prepare().expect("prepare first activation")
        else {
            panic!("fresh fixture cannot already be activated")
        };
        prepared.commit().expect("commit first activation");

        match mutation {
            Mutation::WrongCurrent => {
                std::fs::remove_file(fixture.root.join("current")).expect("remove current link");
                std::os::unix::fs::symlink(
                    "activations/00000000000000000002",
                    fixture.root.join("current"),
                )
                .expect("replace current link");
            }
            Mutation::ExtraActivation => {
                std::fs::create_dir(fixture.root.join("activations/00000000000000000002"))
                    .expect("create extra activation");
            }
            Mutation::MutatedReceipt => {
                let receipt = fixture
                    .root
                    .join("activations/00000000000000000001/install-receipt.json");
                let mut bytes = std::fs::read(&receipt).expect("read receipt");
                bytes.push(b' ');
                std::fs::write(receipt, bytes).expect("mutate receipt");
            }
        }

        let locked = lock_metadata_state(&authorization).expect("lock mutated metadata state");
        assert!(locked.read_live_installed_release_floor().is_err());
    }
}
