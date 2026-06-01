# Download-only sermons missing from the transcript catalog

Sermons in the church podcast feed (`fbcministries.net/feed/podcast`) that have a
downloadable MP3 but **no YouTube video**, so the current YouTube-driven pipeline
never ingested them. Cross-referenced locally against `sermons_catalog.json`.

Generated from a one-time feed crawl (no live re-crawl). The `video_id` column is a
proposed synthetic id (`fbc-` prefix) for ingest. **Note:** some older MP3 links
(esp. pre-2023) currently 404 on the church server; confirm availability one-at-a-
time and gently at ingest, skipping dead links.

- **Genuinely missing:** 294
- **Needs review (resembles an existing transcript):** 41
- **Likely already in catalog (same title + same date):** 0

## Genuinely missing (candidates to add)

### Elijah And Elisha Series

- `2020-03-25` [MW] Naaman The Leper
- `2020-03-11` [MW] Provision For The Sons Of The Prophets
- `2020-02-26` [MW] Provision In Shunem
- `2020-02-19` [MW] Just A Pot Of Oil
- `2020-02-12` [MW] Make This Valley Full Of Ditches
- `2020-02-05` [MW] I Have Healed These Waters
- `2020-01-29` [MW] Elisha Saw It
- `2020-01-22` [MW] Elisha The Son Of Shaphat
- `2020-01-15` [MW] Elijah Went Up By A Whirlwind Into Heaven
- `2020-01-08` [MW] The Fire Of God Consumed Them
- `2019-12-18` [MW] It Is Elijah The Tishbite
- `2019-12-11` [MW] Elijahs Final Encounter With Ahab
- `2019-12-04` [MW] Elijah Requested Death
- `2019-11-06` [MW] A Sound Of Abundance Of Rain
- `2019-10-30` [MW] The Fire Of The Lord Fell
- `2019-10-16` [MW] How Long Halt Ye Between Two Opinions
- `2019-10-09` [MW] Trouble In Israel
- `2019-09-18` [MW] Thy Son Liveth
- `2019-09-11` [MW] Elijah At Zarephath
- `2019-09-04` [MW] The Brook Dried Up
- `2019-08-28` [MW] Elijah At Cherith
- `2019-08-21` [MW] Elijah The Tishbite

### Genesis Series

- `2023-10-29` [PM] I Have Set Thee Over All The Land Of Egypt
- `2023-02-12` [PM] Jacobs Encounter With God
- `2022-08-07` [PM] And Adam Knew His Wife
- `2022-05-15` [PM] God Hath Seen
- `2020-03-01` [PM] The Ice Age
- `2020-01-26` [PM] The Waters Decreased Continually
- `2019-12-15` [PM] Noah Worshipped The Lord
- `2019-12-08` [PM] God Remembered Noah
- `2019-12-01` [PM] Come Into The Ark
- `2019-11-24` [PM] Make Thee An Ark
- `2019-11-10` [PM] Noah Walked With God
- `2019-11-03` [PM] The Wickedness Of Man Was Great
- `2019-10-13` [PM] The Godly Line Of Seth
- `2019-10-06` [PM] The Ungodly Line Of Cain
- `2019-09-15` [PM] Cain Slew Abel
- `2019-08-25` [PM] An Acceptable Offering
- `2019-08-18` [PM] Coats Of Skins
- `2019-08-11` [PM] Gods Judgement Announced
- `2019-07-28` [PM] Hiding From God
- `2019-07-21` [PM] Sin Entered The World
- `2019-07-14` [PM] The Sanctity Of Marriage
- `2019-07-07` [PM] Life In The Garden Of Eden
- `2019-06-30` [PM] The Heavens And The Earth Were Finished
- `2019-06-09` [PM] God Created Man In His Own Image
- `2019-06-02` [PM] Cattle Beasts And Creeping Things
- `2019-04-28` [PM] The Fifth Day
- `2019-04-14` [PM] The Fourth Day
- `2019-03-31` [PM] The Second Day
- `2019-03-24` [PM] The First Day
- `2019-03-17` [PM] God Created
- `2019-03-10` [PM] In The Beginning

### Hebrews Series

- `2020-03-15` [AM] The Throne Of Grace
- `2020-03-08` [AM] The Word Of God Is Quick And Powerful
- `2020-03-01` [AM] Hearing And Believing
- `2020-02-16` [AM] More Glory Than Moses
- `2020-02-09` [AM] Jesus Tasted Death For Every Man
- `2020-01-26` [AM] Christ So Much Better Than The Angels
- `2020-01-19` [AM] Our Glorious Saviour
- `2020-01-12` [AM] Looking Unto Jesus

### Revelation Series

- `2024-12-08` [PM] Unto The Church In Smyrna

### Ruth Series

- `2023-11-15` [MW] Lessons From Ruth
- `2023-11-08` [MW] They Called His Name Obed
- `2023-11-01` [MW] Boaz Completed The Transaction
- `2023-10-25` [MW] I Will Do To Thee All That Thou Requirest
- `2023-09-06` [MW] Handfuls Of Purpose
- `2023-08-30` [MW] Boaz Meets Ruth
- `2023-08-23` [MW] Ruth In The Field Of Boaz
- `2023-08-16` [MW] They Came To Bethlehem
- `2023-08-09` [MW] Ruth Made Her Choice

### Timothy Series

- `2024-12-08` [AM] Profitable For The Ministry
- `2024-11-24` [AM] I Have Finished My Course
- `2024-08-11` [AM] Redemption Through His Blood
- `2024-08-04` [AM] He Oft Refreshed Me

### Titus Series

- `2020-03-15` [SS] Bibliology Part 2
- `2020-03-08` [SS] Bibliology Part 1
- `2020-03-01` [SS] Sound Doctrine
- `2020-02-23` [SS] Rebuke Them Sharply
- `2020-02-16` [SS] Qualifications For Elders Part 2
- `2020-02-09` [SS] Qualifications For Elders
- `2020-02-02` [SS] The Ministry Of Titus In Crete
- `2020-01-26` [SS] In Hope Of Eternal Life

### Walk Worthy

- `2023-11-29` [MW] Overwhelmed And Desolate Part 1

### Walk Worthy Series

- `2024-07-03` [MW] The Totality Of Scripture
- `2024-05-01` [MW] Ponder The Promises

- `2026-05-10` [PM] Thou Hast Put Gladness In My Heart
- `2026-04-01` [MW] By Faith Abel
- `2026-03-25` [MW] Conquering Doubt
- `2026-03-18` [MW] Consequences Of Doubt
- `2026-03-11` [MW] Identifying Doubt
- `2026-02-25` [MW] They Believed Not
- `2026-02-18` [MW] Because Of Your Unbelief
- `2026-02-11` [MW] If Thou Canst Do Anything
- `2026-01-21` [MW] A Longing To See God Work Part Three
- `2026-01-14` [MW] A Longing To See God Work Part Two
- `2026-01-07` [MW] A Longing To See God Work Part One
- `2025-12-28` [PM] Five Tips For A Successful Year
- `2025-12-24` [PM] Christmas Message
- `2025-12-17` [MW] What If They Hearkened
- `2025-12-10` [MW] They Hearkened Not
- `2025-12-03` [MW] Wilt Thou Hearken
- `2025-11-19` [MW] Proved At Meribah
- `2025-11-09` [PM] Back To Calvary
- `2025-11-05` [MW] Went Out Of Egypt
- `2025-10-29` [MW] Praise With Song
- `2025-10-22` [MW] The Feast Of Tabernacles
- `2025-10-15` [MW] A Longing For Forgiveness
- `2025-10-12` [PM] Come And Help
- `2025-10-08` [MW] A Longing For Direction
- `2025-10-01` [MW] A Longing For Peace
- `2025-09-24` [MW] We Will Remember The Name Of Our God
- `2025-09-10` [MW] My Soul Thirsteth For God
- `2025-09-03` [MW] El Olam
- `2025-08-27` [MW] El Roi
- `2025-08-06` [MW] When They Had Fasted And Prayed
- `2025-07-16` [MW] I Will Rain Bread From Heaven
- `2025-07-09` [MW] How Do We Respond To God
- `2025-07-02` [MW] Jehovah Shalom
- `2025-06-25` [MW] Beware Of False Teachers
- `2025-05-21` [MW] Jesus Our Shepherd
- `2025-05-14` [MW] The Presence Of My Shepherd
- `2025-05-07` [MW] The Protection Of My Shepherd
- `2025-04-23` [MW] My Shepherd Provides For Me
- `2025-04-20` [AM] Jesus Drew Near And Went With Them
- `2025-04-16` [MW] Jehovah Raah
- `2025-04-09` [MW] Jehovah Rapha
- `2025-03-26` [MW] Jehovah
- `2025-03-19` [MW] Jehovah Jireh
- `2025-02-26` [MW] Adonai
- `2025-02-19` [MW] El Elyon
- `2025-02-05` [MW] Elohim
- `2025-01-29` [MW] Because He Hath Known My Name
- `2025-01-15` [MW] What Can The Righteous Do
- `2025-01-08` [MW] Under Attack
- `2024-12-22` [PM] Let Me Not Wander From Thy Commandments
- `2024-12-04` [MW] The Declaration Of The Psalmist
- `2024-09-01` [PM] Consecrated For Service
- `2023-10-18` [MW] Why Scripture Assembly
- `2023-09-24` [PM] They Beckoned Unto Their Partners
- `2023-09-13` [MW] Revive Us Again
- `2023-04-12` [MW] The Lord Is My Strong Refuge
- `2023-04-05` [MW] The Lord Is My Hope
- `2023-03-29` [MW] The Lord Is My Fortress
- `2023-03-23` [MW] The Lord Is My Rock
- `2023-03-01` [MW] The End Is Peace
- `2023-02-15` [MW] Our Divine Protector
- `2023-02-08` [MW] The Law Of His God Is In His Heart
- `2023-02-01` [MW] The Lord Forsaketh Not His Saints
- `2023-01-25` [MW] A Recognition Of Gods Faithfulness
- `2023-01-18` [MW] The Lord Upholdeth Him With His Hand
- `2023-01-15` [PM] Conviction Or Preference
- `2023-01-11` [MW] The Generosity Of The Righteous
- `2023-01-08` [PM] When Ye Fast
- `2023-01-04` [MW] The Lord Knoweth Your Days
- `2022-12-28` [MW] Preparation For A New Year
- `2022-12-25` [AM] They Presented Unto Him Gifts
- `2022-12-21` [MW] The Lord Upholdeth The Righteous
- `2022-12-18` [PM] They Shall Call His Name Emmanuel
- `2022-12-14` [MW] When Less Is More
- `2022-12-07` [MW] The Hostility Of The Wicked
- `2022-11-30` [MW] Cease From Anger
- `2022-11-27` [PM] Wherefore Then Serveth The Law
- `2022-11-16` [MW] Rest In The Lord
- `2022-10-12` [MW] Trust In The Lord And Do Good
- `2022-10-05` [MW] Fret Not Thyself Because Of Evildoers
- `2022-09-14` [MW] The Power Of II Chronicles 7:14
- `2022-09-07` [MW] The Actions Of II Chronicles 7:14
- `2022-08-31` [MW] The Condition Of II Chronicles 7:14
- `2022-08-28` [PM] The Indwelling Of The Holy Spirit
- `2022-08-24` [MW] Grace
- `2022-08-17` [MW] My Fellowlabourers
- `2022-08-14` [PM] What Does The Bible Say About Ghosts
- `2022-08-10` [MW] The Power Of Prayer
- `2022-08-03` [MW] Confident In Thy Obedience
- `2022-07-24` [PM] The Lords Throne Is In Heaven
- `2022-07-20` [MW] Put That On Mine Account
- `2022-07-17` [PM] And Upon The First Day Of The Week
- `2022-07-06` [MW] Yet For Loves Sake
- `2022-06-19` [PM] Hear Learn Do
- `2022-06-05` [PM] What Does The Bible Say About Purgatory
- `2022-06-01` [MW] Philemon Our Dearly Beloved
- `2022-05-18` [MW] For The Gospels Sake
- `2022-05-04` [MW] Lessons In The Valley Of Rephaim
- `2022-04-24` [PM] Walk In Newness Of Life
- `2022-04-20` [MW] The Growth Of The Thessalonians
- `2022-04-06` [MW] Learning From Others Mistakes
- `2022-03-30` [MW] Learning From Your Mistakes
- `2022-03-16` [MW] The Classroom Of Loneliness
- `2022-02-16` [MW] Tips To Promote Learning
- `2022-01-19` [MW] Teach Me
- `2021-12-19` [PM] Simeon Was Waiting For The Promised Messiah
- `2021-09-15` [MW] The Lord Is With Him
- `2020-03-29` [AM] Our God Answers Prayer
- `2020-03-29` [PM] Acknowledge Thine Iniquity
- `2020-03-22` [AM] He Will Deliver Thee
- `2020-03-18` [MW] The Blessing Of Peace
- `2020-03-15` [PM] Draw Near To God
- `2020-03-08` [PM] Diminish Not A Word
- `2020-03-04` [MW] Take Up Thy Son
- `2020-02-23` [PM] He First Loved Us
- `2020-02-16` [PM] Publish Glad Tidings
- `2020-02-09` [PM] Are You Overwhelmed
- `2020-02-02` [PM] The Coming Of The Lord Draweth Nigh
- `2020-01-19` [SS] Grounded And Settled
- `2020-01-12` [PM] 2020 Vision
- `2019-12-29` [PM] I Have Finished My Course
- `2019-12-29` [AM] Run Your Race
- `2019-12-22` [PM] Fell Down And Worshipped Him
- `2019-12-22` [AM] The Word Was Made Flesh
- `2019-12-22` [SS] The Prince Of Peace
- `2019-12-15` [AM] The Ministry Of Tychicus
- `2019-12-08` [AM] Make Known The Gospel
- `2019-12-01` [AM] Put On The Whole Armor Of God
- `2019-12-01` [SS] What Hast Thou That Thou Didst Not Receive
- `2019-11-24` [SS] Caleb Followed The Lord
- `2019-11-13` [MW] Jobs Ministry To Others
- `2019-11-10` [AM] Prepare For Battle
- `2019-11-10` [SS] My Tongue Shall Speak Of Thy Word
- `2019-11-03` [AM] A Godly Testimony In The Workplace
- `2019-11-03` [SS] Great Peace Have They Which Love Thy Law
- `2019-10-23` [MW] Fellowlabourers
- `2019-10-20` [PM] Be Still
- `2019-10-20` [AM] Especially The Parchments
- `2019-10-20` [SS] A Plea For Deliverance
- `2019-10-13` [AM] A Glorious Church
- `2019-10-13` [SS] All Thy Commandments Are Truth
- `2019-10-06` [AM] Revival In Your Home
- `2019-10-06` [SS] Thy Word Is Very Pure
- `2019-09-22` [SS] Thy Testimonies Are Wonderful
- `2019-09-15` [AM] Filled With The Spirit
- `2019-09-15` [SS] A Desire To See God Work
- `2019-09-08` [AM] Walk Circumspectly
- `2019-09-08` [SS] Thou Art My Hiding Place
- `2019-09-01` [AM] Walk As Children Of Light
- `2019-09-01` [SS] Thy Word Is A Lamp Unto My Feet
- `2019-08-25` [AM] Walk In Love
- `2019-08-18` [AM] Be Ye Kind
- `2019-08-18` [SS] The Bible Stands
- `2019-08-11` [AM] Speech That Edifies
- `2019-08-11` [SS] When Wilt Thou Comfort Me
- `2019-08-07` [MW] Some Things To Remember In Times Of Heaviness
- `2019-08-04` [AM] The Value Of Work
- `2019-08-04` [SS] Let My Heart Be Sound In Thy Statutes
- `2019-07-31` [MW] Laboring In Prayer
- `2019-07-28` [AM] The Word Of The Lord Endureth Forever
- `2019-07-28` [SS] Lessons Through Affliction
- `2019-07-21` [SS] Allowing His Word To Direct My Ways
- `2019-07-17` [MW] The Wind Was Contrary
- `2019-07-14` [AM] Give No Place To The Devil
- `2019-07-14` [SS] Comfort In Affliction
- `2019-07-10` [MW] Lift Up Your Eyes
- `2019-07-07` [AM] Be Ye Angry And Sin Not
- `2019-07-07` [SS] I Trust In Thy Word
- `2019-07-03` [MW] He Will Help You
- `2019-06-30` [AM] Speak Every Man Truth
- `2019-06-30` [SS] Lifelong Obedience
- `2019-06-23` [AM] Im Different Now
- `2019-06-23` [SS] Cling To The Word
- `2019-06-19` [MW] He Is Able To Succour Them That Are Tempted
- `2019-06-16` [AM] David Charged Solomon His Son
- `2019-06-16` [SS] Open Thou Mine Eyes
- `2019-06-12` [MW] Having Obtained Help Of God
- `2019-06-09` [AM] The Word Of God Will Build You Up
- `2019-06-09` [SS] Let Me Not Wander From Thy Commandments
- `2019-06-05` [MW] Have Mercy On Me
- `2019-05-29` [MW] Our God Will Help
- `2019-05-26` [SS] Commanded To Obey
- `2019-05-19` [PM] Bring Them In
- `2019-05-12` [AM] She Hid Him Three Months
- `2019-05-12` [SS] How Shall They Hear
- `2019-05-05` [PM] The Harvest Is Plenteous
- `2019-05-01` [MW] God Hath Power To Help
- `2019-04-28` [SS] The Salutation Of Paul
- `2019-04-24` [MW] Thou Wilt Hear And Help
- `2019-04-21` [PM] Thy Greatness
- `2019-04-21` [AM] Preached Through Jesus The Resurrection From The Dead
- `2019-04-17` [MW] It Is Nothing With Thee To Help
- `2019-04-14` [AM] Truly This Man Was The Son Of God
- `2019-04-14` [SS] Greetings From The Brethren
- `2019-04-10` [MW] I Am Helped
- `2019-04-07` [PM] The Third Day
- `2019-04-07` [SS] A Final Charge
- `2019-04-03` [MW] The Lord Helped Us
- `2019-03-31` [AM] Unto Him That Is Able
- `2019-03-31` [SS] Partners In The Ministry
- `2019-03-27` [MW] The Lord Is My Helper
- `2019-03-24` [AM] The Unsearchable Riches Of Christ
- `2019-03-24` [SS] Open Doors
- `2019-03-20` [MW] It Is Worth It
- `2019-03-17` [AM] The Household Of God
- `2019-03-17` [SS] The Collection For The Saints
- `2019-03-13` [MW] That Ye May Grow
- `2017-01-08` [PM] The Deceitfulness Of Alcohol

## Needs review

### Doers Of The Word Series

- `2026-01-18` [AM] We Lack Wisdom  — ~1.00 vs `"We Lack Wisdom"  James 1:5-8`

### Doing What We Ought Series

- `2024-11-20` [MW] We Ought To Walk As Christ Walked Part 5  — ~0.71 vs `"We Ought To Walk As Christ Walked - Part 4" I John 2:6`

### Genesis Series

- `2024-04-07` [PM] Jacob Blesses His Sons Part 3  — ~0.80 vs `"Jacob Blesses His Sons - Part Two" Genesis 49:13-19`

### Gods Grace Is Sufficient Series

- `2025-12-28` [AM] Gods Grace Is Sufficient Part Two  — ~0.50 vs `"God's Grace Is Sufficient - Part Two" II Corinthians 12:9`

### Great Is The Lord Series

- `2023-12-03` [AM] So Run That Ye May Obtain  — ~1.00 vs `"So Run, That Ye May Obtain" Acts 20:17-36`

### Hebrews Series

- `2020-10-04` [AM] Living By Faith Part 3  — ~0.67 vs `"Living By Faith - Part 2" James 2:25-26`
- `2020-02-23` [AM] Are You Living In The Wilderness  — ~0.60 vs `"Who Are You Living For?" Philippians 1:21`
- `2020-02-02` [AM] Listen Up  — ~0.50 vs `"We Ought To Listen Up - Part 2" Hebrews 2:1-4`

### Timothy Series

- `2024-11-10` [AM] Preparation For Persecution  — ~0.67 SAME-DATE vs `"Preparation For Persecution" II Timothy 3:10-12`

### Walk Worthy Series

- `2024-03-20` [MW] In Every Thing Give Thanks  — ~0.80 vs `"In Every Thing Give Thanks" I Thessalonians 5:18`
- `2024-02-28` [MW] Rejoice Evermore  — ~0.67 vs `"Rejoice Evermore" I Thessalonians 5:16`

- `2026-05-20` [MW] Faith Through Trials  — ~1.00 vs `"Faith Through Trials"  Hebrews 11:17-19`
- `2026-05-13` [MW] Through Faith Sara  — ~0.50 vs `"Faith Through Trials"  Hebrews 11:17-19`
- `2025-11-12` [MW] Thou Called I Answered  — ~0.50 vs `"Thou Called And I Delivered" Psalm 81:7`
- `2025-06-04` [MW] Why Have A Missions Conference  — ~0.50 vs `"Why Missions?" Romans 10:13-17`
- `2025-03-12` [MW] El Shaddai  — ~1.00 vs `"El Shaddai" Genesis 17:1`
- `2025-01-22` [MW] The Righteous Lord Loveth Righteousness  — ~1.00 vs `"The Righteous Lord Loveth Righteousness" Psalm 11:7`
- `2024-08-25` [PM] Tongues Shall Cease  — ~0.67 vs `"Tongues Shall Cease" I Corinthians 13:8`
- `2024-07-14` [PM] Fishers Of Men Part 3  — ~0.75 vs `"Fishers Of Men - Part Three" Matthew 4:19`
- `2023-03-15` [MW] The Lord Is My ...  — ~0.50 vs `"Great Is The Lord" Psalm 48:1-2`
- `2023-01-01` [PM] Living Your Faith  — ~1.00 vs `"Living By Faith - Part 2" James 2:25-26`
- `2022-12-11` [PM] This Do In Remembrance Of Me  — ~0.75 vs `"This Do In Remembrance Of Me" I Corinthians 11:23-32`
- `2022-09-11` [PM] His Church  — ~0.50 vs `"A Praying Church"  Acts 12:1-19`
- `2022-05-11` [MW] In The Lord Put I My Trust  — ~0.50 vs `"I The Lord"  Isaiah 41:17-20`
- `2022-04-13` [MW] Teach Others Also  — ~0.75 vs `"Teach Others Also" II Timothy 2:1-2`
- `2021-11-17` [MW] Deceiving Your Own Selves  — ~1.00 vs `"Deceiving Your Own Selves" James 1:22`
- `2019-12-29` [SS] A Good Soldier  — ~0.67 vs `"A Good Soldier - Part 5" II Timothy 2:6, 24-26`
- `2019-11-24` [AM] Worthy Of Praise  — ~0.67 vs `"He Is Worthy To Receive Praise"  Revelation 4:10-11`
- `2019-09-25` [MW] Break Up Your Fallow Ground  — ~1.00 vs `"Break Up Your Fallow Ground"  Hosea 10`
- `2019-09-22` [AM] Yield Yourselves Unto The Lord  — ~0.50 vs `"Yield Yourselves Unto God" Romans 6:1-19`
- `2019-09-01` [PM] He Hath Dealt Bountifully With Me  — ~0.50 vs `"God Hath Dealt Graciously With Me" Genesis 33:1-20`
- `2019-08-25` [SS] O How Love I Thy Law  — ~1.00 vs `"O How Love I Thy Law" Psalm 119:97`
- `2019-07-21` [AM] I Saw The Lord  — ~0.67 vs `"I The Lord"  Isaiah 41:17-20`
- `2019-06-16` [PM] God Hath Dealt Graciously With Me  — ~1.00 vs `"God Hath Dealt Graciously With Me" Genesis 33:1-20`
- `2019-05-26` [PM] This Do In Remembrance Of Me  — ~0.75 vs `"This Do In Remembrance Of Me" I Corinthians 11:23-32`
- `2019-05-26` [AM] These Stones Shall Be A Memorial  — ~1.00 vs `"These Stones Shall Be For A Memorial" Joshua 4:1-24`
- `2019-05-05` [AM] We Are His Witnesses  — ~0.50 vs `"We Are His Workmanship" Ephesians 2:10`
- `2019-05-05` [SS] Every Where Preaching The Word  — ~0.80 vs `"Went Every Where Preaching The Word" Acts 8:1-4`
- `2019-04-28` [AM] Are You Ready  — ~0.75 vs `"Are You Ready?" II Samuel 15:15`
- `2019-04-21` [SS] Nailing It To His Cross  — ~1.00 vs `"Nailing It To His Cross" Colossians 2:11-15`
- `2019-04-07` [AM] Walk Worthy  — ~0.67 vs `"Walk Worthy" I Thessalonians 2:12`

## Likely already in catalog

