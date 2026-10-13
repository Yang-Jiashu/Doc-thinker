"""Local owner-only administration. No HTTP admin route or automatic worker start."""
import argparse
import json
import os
from pathlib import Path

from .invite_store import InviteError, InviteStore, _private_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Manage a private invitation-only DocThinker instance")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    add = commands.add_parser("add-user")
    add.add_argument("--username", required=True)
    add.add_argument("--upstream-port", type=int, required=True)
    add.add_argument("--upstream-token-file", type=Path)
    add.add_argument("--expires-hours", type=int, default=48)
    add.add_argument("--allow-uploads", action="store_true", help="Requires an explicitly enabled research profile (or separately configured legacy upload profile)")
    revoke = commands.add_parser("revoke-user")
    revoke.add_argument("--username", required=True)
    reissue = commands.add_parser("reissue-invite", help="Replace a pending user's code; every successful call invalidates its previous codes")
    reissue.add_argument("--username", required=True)
    reissue.add_argument("--expires-hours", type=int, default=48)
    export = commands.add_parser("export-account", help="Cold export of one revoked new-protocol account; no credentials")
    export.add_argument("--username", required=True)
    export.add_argument("--confirm-username", required=True)
    export.add_argument("--output", type=Path, required=True)
    purge = commands.add_parser("purge-account", help="Cold retained-data cleanup, not identity/backup/physical secure erase")
    purge.add_argument("--username", required=True)
    purge.add_argument("--confirm-username", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            directory = os.getenv("DOCTHINKER_INVITE_DATA_DIR", "").strip()
            if not directory:
                raise InviteError("Set an explicit DOCTHINKER_INVITE_DATA_DIR first")
            InviteStore.initialize(directory)
            print("Invitation store initialized. No workers were started.")
            return 0
        store = InviteStore.from_environment()
        if args.command == "add-user":
            token = None
            if args.upstream_token_file:
                filename = args.upstream_token_file
                if not filename.is_absolute():
                    raise InviteError("Upstream token file must be absolute")
                _private_path(filename, directory=False)
                if filename.stat().st_size > 512:
                    raise InviteError("Invalid upstream token file")
                token = filename.read_text(encoding="utf-8").strip()
            result = store.add_user(args.username, args.upstream_port, upstream_token=token,
                                    expires_hours=args.expires_hours, allow_uploads=args.allow_uploads)
            # This is the sole plaintext invitation output. Never log it or put it
            # into a URL; the operator passes it privately to the intended person.
            print(json.dumps(result, ensure_ascii=False))
        elif args.command == "reissue-invite":
            # Same one-time owner-only output as add-user; never include an
            # old code, browser token, worker token or another user's identity.
            print(json.dumps(store.reissue_invitation(args.username, expires_hours=args.expires_hours), ensure_ascii=False))
        elif args.command in {"export-account", "purge-account"}:
            from .invite_account_lifecycle import export_account, purge_account

            if args.command == "export-account":
                result = export_account(store, args.username, args.confirm_username, args.output)
            else:
                result = purge_account(store, args.username, args.confirm_username)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        else:
            changed = store.revoke_user(args.username)
            print("User and credentials revoked." if changed else "No matching user.")
        return 0
    except InviteError as error:
        print(str(error), file=__import__("sys").stderr)
    except Exception:
        print("Invitation administration failed; no credential details are displayed.", file=__import__("sys").stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
